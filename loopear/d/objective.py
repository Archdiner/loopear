import torch

from loopear.d.spec import FREQ_MASK_BANDS, GRID_F, GRID_T, K1_PRIOR, TIME_MASK_RATIO


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
