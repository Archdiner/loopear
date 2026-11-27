import torch

from loopear.d.spec import FREQ_MASK_BANDS, GRID_F, GRID_T, K1_PRIOR, PATCH_FRAMES, TIME_MASK_RATIO

TIME_NEIGHBORHOOD_PATCHES: int = 1


def global_ntxent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    if z1.ndim != 2 or z2.shape != z1.shape:
        raise ValueError(f"expected matching [B, D] views, got {tuple(z1.shape)} {tuple(z2.shape)}")
    if z1.shape[0] < 2:
        raise ValueError("global contrast needs at least 2 recordings")
    b = z1.shape[0]
    eye = torch.eye(b, dtype=torch.bool, device=z1.device)
    sim_11 = z1 @ z1.T / tau
    sim_22 = z2 @ z2.T / tau
    sim_12 = z1 @ z2.T / tau
    sim_21 = z2 @ z1.T / tau
    sim_11 = sim_11.masked_fill(eye, -1e9)
    sim_22 = sim_22.masked_fill(eye, -1e9)
    loss_1 = -(sim_12.diag() - torch.logsumexp(torch.cat([sim_11, sim_12], dim=1), dim=1))
    loss_2 = -(sim_21.diag() - torch.logsumexp(torch.cat([sim_22, sim_21], dim=1), dim=1))
    return (loss_1.mean() + loss_2.mean()) * 0.5


def _shift_time(x: torch.Tensor, delta: int) -> torch.Tensor:
    if delta == 0:
        return x
    out = torch.zeros_like(x)
    if delta > 0:
        out[:, : x.shape[1] - delta] = x[:, delta:]
    else:
        out[:, -delta:] = x[:, : x.shape[1] + delta]
    return out


def _nce_positions(
    pred: torch.Tensor,
    target: torch.Tensor,
    anchor_ok: torch.Tensor,
    target_ok: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, f, d = pred.shape
    p = t * f
    pr = pred.reshape(b, p, d)
    tg = target.reshape(b, p, d)
    aok = anchor_ok.reshape(b, p)
    tok = target_ok.reshape(b, p)
    logits = torch.einsum("bpd,jpd->pbj", pr, tg) / tau
    logits = logits.masked_fill(~tok.transpose(0, 1)[:, None, :], -1e9)
    logz = torch.logsumexp(logits, dim=-1)
    pos = (pr * tg).sum(dim=-1).transpose(0, 1) / tau
    loss = -(pos - logz)
    enough = tok.sum(dim=0) >= 2
    use = aok.transpose(0, 1) & tok.transpose(0, 1) & enough[:, None]
    return loss.transpose(0, 1).reshape(b, t, f), use.transpose(0, 1).reshape(b, t, f)


def local_mld(
    pred: torch.Tensor,
    target: torch.Tensor,
    anchor_mask: torch.Tensor,
    valid: torch.Tensor,
    tau: float,
    k: int,
) -> torch.Tensor:
    if k < 0:
        raise ValueError("k must be non-negative")
    if pred.shape != target.shape:
        raise ValueError("pred and target shapes differ")
    if anchor_mask.shape != valid.shape or anchor_mask.shape != pred.shape[:3]:
        raise ValueError("mask shapes do not match the grid")
    if k == 0:
        priors = {0: 1.0}
    elif k == 1:
        priors = {-1: K1_PRIOR[0], 0: K1_PRIOR[1], 1: K1_PRIOR[2]}
    else:
        raise ValueError("only K=0 and K=1 are implemented")
    anchor_base = anchor_mask & valid
    weighted = torch.zeros(pred.shape[:3], device=pred.device, dtype=pred.dtype)
    weight_sum = torch.zeros_like(weighted)
    for delta, prior in priors.items():
        shifted_target = _shift_time(target, delta)
        shifted_valid = _shift_time(valid, delta)
        shifted_mask = _shift_time(anchor_mask, delta)
        eligible = shifted_mask & shifted_valid
        anchor_ok = anchor_base & eligible
        loss, use = _nce_positions(pred, shifted_target, anchor_ok, shifted_valid, tau)
        weighted = weighted + torch.where(use, loss, torch.zeros_like(loss)) * float(prior)
        weight_sum = weight_sum + use.to(pred.dtype) * float(prior)
    active = weight_sum > 0
    if not bool(active.any()):
        return pred.sum() * 0.0
    per = torch.where(active, weighted / weight_sum, torch.zeros_like(weighted))
    return per[active].mean()


def sample_time_mask(valid_time: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if valid_time.ndim != 2 or valid_time.shape[1] != GRID_T:
        raise ValueError(f"valid_time must be [B, {GRID_T}]")
    b, t = valid_time.shape
    mask = torch.zeros(b, t, dtype=torch.bool)
    for row in range(b):
        idx = torch.nonzero(valid_time[row], as_tuple=False).flatten()
        n = int(idx.numel())
        if n == 0:
            continue
        k = int(round(TIME_MASK_RATIO * n))
        k = min(max(k, 1), n)
        pick = idx[torch.randperm(n, generator=generator)[:k]]
        mask[row, pick] = True
    return mask


def sample_freq_mask(batch: int, generator: torch.Generator) -> torch.Tensor:
    k = min(max(FREQ_MASK_BANDS, 1), GRID_F)
    mask = torch.zeros(batch, GRID_F, dtype=torch.bool)
    for row in range(batch):
        pick = torch.randperm(GRID_F, generator=generator)[:k]
        mask[row, pick] = True
    return mask


def view_masks(
    pad: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    if pad.shape[1:] != (GRID_T, GRID_F):
        raise ValueError(f"pad must be [B, {GRID_T}, {GRID_F}]")
    valid_time = ~pad.all(dim=-1)
    time_mask = sample_time_mask(valid_time, generator)
    freq_mask = sample_freq_mask(pad.shape[0], generator)
    masked_time = time_mask[:, :, None].expand_as(pad) & ~pad
    masked_freq = freq_mask[:, None, :].expand_as(pad) & ~pad
    return masked_time, masked_freq


def patch_valid_from_frames(frame_valid: torch.Tensor) -> torch.Tensor:
    if frame_valid.ndim != 2 or frame_valid.shape[1] != GRID_T * 4:
        raise ValueError("frame_valid must be [B, 200]")
    b = frame_valid.shape[0]
    grouped = frame_valid.view(b, GRID_T, 4)
    return grouped.all(dim=-1)


def broadcast_time_valid(time_valid: torch.Tensor) -> torch.Tensor:
    return time_valid[:, :, None].expand(-1, -1, GRID_F).contiguous()


def crop_pair_shift(start_a: int, start_b: int) -> int | None:
    delta = int(start_a) - int(start_b)
    if delta % PATCH_FRAMES != 0:
        return None
    return delta // PATCH_FRAMES


def _shift_rows(pos_time: torch.Tensor, shift: int) -> torch.Tensor:
    if pos_time.ndim != 2:
        raise ValueError("time embeddings must be [T, D]")
    n_time = int(pos_time.shape[0])
    if shift == 0 or abs(shift) >= n_time:
        return torch.zeros(0, dtype=torch.bool, device=pos_time.device)
    index = torch.arange(n_time, device=pos_time.device)
    other = index + int(shift)
    keep = (other >= 0) & (other < n_time)
    same = torch.all(pos_time[index[keep]] == pos_time[other[keep]], dim=-1)
    return same


def shared_time_embedding(pos_tables: tuple[torch.Tensor, ...], shift: int) -> bool:
    for table in pos_tables:
        if bool(_shift_rows(table, shift).any()):
            return True
    return False


def describe_crop_offsets(
    start1: torch.Tensor,
    start2: torch.Tensor,
    pos_tables: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    if start1.shape != start2.shape or start1.ndim != 1:
        raise ValueError("crop starts must be [B]")
    n_same = 0
    n_unaligned = 0
    n_aligned = 0
    n_pairs = 0
    n_shared = 0
    n_distinct = 0
    for start_a, start_b in zip(start1.tolist(), start2.tolist()):
        shift = crop_pair_shift(int(start_a), int(start_b))
        if shift is None:
            n_unaligned += 1
            continue
        if shift == 0:
            n_same += 1
            continue
        n_aligned += 1
        n_time = int(pos_tables[0].shape[0])
        reach = max(0, n_time - abs(int(shift)))
        n_pairs += reach
        if shared_time_embedding(pos_tables, int(shift)):
            n_shared += reach
        else:
            n_distinct += reach
    return {
        "n_recordings": float(start1.shape[0]),
        "n_same_offset": float(n_same),
        "n_unaligned": float(n_unaligned),
        "n_aligned_recordings": float(n_aligned),
        "n_index_pairs": float(n_pairs),
        "n_shared_embedding_pairs": float(n_shared),
        "n_distinct_embedding_pairs": float(n_distinct),
    }


def _align_time(value: torch.Tensor, shift: int) -> torch.Tensor:
    out = torch.zeros_like(value)
    n_time = int(value.shape[1])
    if shift >= n_time or shift <= -n_time:
        return out
    if shift >= 0:
        out[:, : n_time - shift] = value[:, shift:]
    else:
        out[:, -shift:] = value[:, : n_time + shift]
    return out


def time_correspondence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_pred: torch.Tensor,
    valid_target: torch.Tensor,
    start_pred: torch.Tensor,
    start_target: torch.Tensor,
    pos_tables: tuple[torch.Tensor, ...],
    tau: float,
    neighborhood: int = TIME_NEIGHBORHOOD_PATCHES,
) -> tuple[torch.Tensor, int]:
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("pred and target grids must match")
    if valid_pred.shape != pred.shape[:3] or valid_target.shape != pred.shape[:3]:
        raise ValueError("validity grids must match the predictions")
    if int(neighborhood) < 1:
        raise ValueError("the temporal neighborhood is at least one patch on each side")
    if float(tau) <= 0.0:
        raise ValueError("temperature must be positive")
    n_time = int(pred.shape[1])
    eye = torch.arange(n_time)
    distance = (eye[:, None] - eye[None, :]).abs()
    negatives = distance > int(neighborhood)
    grouped: dict[int, list[int]] = {}
    for row, (start_a, start_b) in enumerate(zip(start_pred.tolist(), start_target.tolist())):
        shift = crop_pair_shift(int(start_a), int(start_b))
        if shift is None or shift == 0 or abs(int(shift)) >= n_time:
            continue
        grouped.setdefault(int(shift), []).append(row)
    loss_sum = pred.new_zeros(())
    count = 0
    for shift, rows in grouped.items():
        for table in pos_tables:
            if bool(_shift_rows(table, shift).any()):
                raise RuntimeError("aligned moments share a time positional embedding")
        chosen = torch.tensor(rows, dtype=torch.long)
        pred_g = pred[chosen]
        aligned = _align_time(target[chosen], shift)
        valid_aligned = _align_time(valid_target[chosen], shift)
        query_ok = valid_pred[chosen] & valid_aligned
        sim = torch.einsum("btfd,bsfd->btsf", pred_g, aligned) / float(tau)
        neg_ok = (
            valid_aligned[:, None, :, :]
            & query_ok[:, :, None, :]
            & negatives[None, :, :, None]
        )
        held = sim.masked_fill(~neg_ok, -1e9)
        logz = torch.logsumexp(held, dim=2)
        positive = (pred_g * aligned).sum(dim=-1) / float(tau)
        use = query_ok & neg_ok.any(dim=2)
        if not bool(use.any()):
            continue
        piece = -(positive - logz)
        loss_sum = loss_sum + piece[use].sum()
        count += int(use.sum())
    if count == 0:
        return pred.sum() * 0.0, 0
    return loss_sum / float(count), count


def temporal_retrieval_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_pred: torch.Tensor,
    valid_target: torch.Tensor,
    start_pred: torch.Tensor,
    start_target: torch.Tensor,
    neighborhood: int = TIME_NEIGHBORHOOD_PATCHES,
) -> dict[str, float]:
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("pred and target grids must match")
    n_time = int(pred.shape[1])
    eye = torch.arange(n_time, device=pred.device)
    distance = (eye[:, None] - eye[None, :]).abs()
    negatives = distance > int(neighborhood)
    positive_key = distance == 0
    totals = {
        "n_queries": 0.0,
        "n_strictly_best": 0.0,
        "n_tied_for_best": 0.0,
        "n_worse": 0.0,
        "rank_sum": 0.0,
        "candidate_sum": 0.0,
        "chance_top1_sum": 0.0,
        "chance_rank_sum": 0.0,
        "correct_cosine_sum": 0.0,
        "n_same_index": 0.0,
        "n_correct_beats_same_index": 0.0,
        "same_index_cosine_sum": 0.0,
        "n_cross_same_index": 0.0,
        "n_correct_beats_cross_same_index": 0.0,
        "cross_same_index_cosine_sum": 0.0,
    }
    other_target = torch.roll(target, 1, dims=0) if int(pred.shape[0]) >= 2 else None
    other_valid = torch.roll(valid_target, 1, dims=0) if other_target is not None else None
    grouped: dict[int, list[int]] = {}
    for row, (start_a, start_b) in enumerate(zip(start_pred.tolist(), start_target.tolist())):
        shift = crop_pair_shift(int(start_a), int(start_b))
        if shift is None or shift == 0 or abs(int(shift)) >= n_time:
            continue
        grouped.setdefault(int(shift), []).append(row)
    for shift, rows in grouped.items():
        chosen = torch.tensor(rows, dtype=torch.long)
        pred_g = pred[chosen]
        aligned = _align_time(target[chosen], shift)
        valid_aligned = _align_time(valid_target[chosen], shift)
        query_ok = valid_pred[chosen] & valid_aligned
        sim = torch.einsum("btfd,bsfd->btsf", pred_g, aligned)
        include = (negatives | positive_key)[None, :, :, None] & valid_aligned[:, None, :, :] & query_ok[:, :, None, :]
        positive = (pred_g * aligned).sum(dim=-1)
        higher = include & (sim > positive[:, :, None, :])
        equal_other = include & positive_key.logical_not()[None, :, :, None] & (sim == positive[:, :, None, :])
        n_higher = higher.sum(dim=2)
        n_cand = include.sum(dim=2)
        use = query_ok & (n_cand >= 2)
        if not bool(use.any()):
            continue
        strictly = use & (n_higher == 0) & (equal_other.sum(dim=2) == 0)
        tied = use & (n_higher == 0) & (equal_other.sum(dim=2) > 0)
        worse = use & (n_higher > 0)
        totals["n_queries"] += float(use.sum())
        totals["n_strictly_best"] += float(strictly.sum())
        totals["n_tied_for_best"] += float(tied.sum())
        totals["n_worse"] += float(worse.sum())
        safe = n_cand.clamp(min=1).to(sim.dtype)
        totals["rank_sum"] += float((1.0 + n_higher.to(sim.dtype))[use].sum())
        totals["candidate_sum"] += float(safe[use].sum())
        totals["chance_top1_sum"] += float((1.0 / safe)[use].sum())
        totals["chance_rank_sum"] += float(((safe + 1.0) * 0.5)[use].sum())
        totals["correct_cosine_sum"] += float(positive[use].sum())
        same = (pred_g * target[chosen]).sum(dim=-1)
        same_ok = use & valid_target[chosen]
        totals["n_same_index"] += float(same_ok.sum())
        totals["n_correct_beats_same_index"] += float((same_ok & (positive > same)).sum())
        totals["same_index_cosine_sum"] += float(same[same_ok].sum())
        if other_target is not None:
            cross = (pred_g * other_target[chosen]).sum(dim=-1)
            cross_ok = use & other_valid[chosen]
            totals["n_cross_same_index"] += float(cross_ok.sum())
            totals["n_correct_beats_cross_same_index"] += float((cross_ok & (positive > cross)).sum())
            totals["cross_same_index_cosine_sum"] += float(cross[cross_ok].sum())
    return totals
