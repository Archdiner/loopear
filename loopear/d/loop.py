import random
from typing import Any

import numpy as np
import torch

from loopear.d.model import LoopEarDSSL
from loopear.d.objective import (
    broadcast_time_valid,
    global_ntxent,
    local_mld,
    patch_valid_from_frames,
    sample_freq_mask,
    sample_time_mask,
)
from loopear.d.spec import GRAD_CLIP, LAMBDA_LOCAL, TAU_GLOBAL, TAU_LOCAL, ema_at, lr_at
from loopear.losses import set_seed


def objective_flags(name: str) -> tuple[float, int]:
    if name == "global":
        return 0.0, 0
    if name == "center":
        return LAMBDA_LOCAL, 0
    if name == "k1":
        return LAMBDA_LOCAL, 1
    raise ValueError(f"unknown objective {name}")


def make_optimizer(model: LoopEarDSSL, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.student_parameters(),
        lr=lr,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    )


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def forward_batch(
    model: LoopEarDSSL,
    mel1: torch.Tensor,
    mel2: torch.Tensor,
    frame_valid1: torch.Tensor,
    frame_valid2: torch.Tensor,
    generator: torch.Generator,
    objective: str,
) -> dict[str, Any]:
    lambda_local, k = objective_flags(objective)
    valid1 = broadcast_time_valid(patch_valid_from_frames(frame_valid1))
    valid2 = broadcast_time_valid(patch_valid_from_frames(frame_valid2))
    if bool((valid1.flatten(1).sum(dim=1) == 0).any()) or bool(
        (valid2.flatten(1).sum(dim=1) == 0).any()
    ):
        raise ValueError("a crop has no valid patches")
    pad1 = ~valid1
    pad2 = ~valid2
    mask1 = sample_time_mask(valid1.any(dim=-1), generator)
    mask1 = mask1[:, :, None].expand_as(valid1) & valid1
    mask2 = sample_freq_mask(mel2.shape[0], generator)
    mask2 = mask2[:, None, :].expand_as(valid2) & valid2
    grid1, cls1 = model.encode_student(mel1, ~mask1 & valid1)
    grid2, cls2 = model.encode_student(mel2, ~mask2 & valid2)
    z1 = model.project_global(cls1)
    z2 = model.project_global(cls2)
    loss_g = global_ntxent(z1, z2, TAU_GLOBAL)
    if lambda_local == 0.0:
        loss_l = loss_g.new_zeros(())
        loss = loss_g
    else:
        pred1 = model.project_local_student(model.predictor(grid1, mask1, pad1, 0))
        pred2 = model.project_local_student(model.predictor(grid2, mask2, pad2, 1))
        target1 = model.encode_teacher(mel1, valid1)
        target2 = model.encode_teacher(mel2, valid2)
        loss_l = 0.5 * (
            local_mld(pred1, target1, mask1, valid1, TAU_LOCAL, k)
            + local_mld(pred2, target2, mask2, valid2, TAU_LOCAL, k)
        )
        loss = loss_g + float(lambda_local) * loss_l
    return {
        "loss": loss,
        "loss_g": loss_g,
        "loss_l": loss_l,
        "z1": z1,
        "z2": z2,
        "k": k,
        "lambda_local": lambda_local,
        "mask1": mask1,
        "mask2": mask2,
        "valid1": valid1,
        "valid2": valid2,
        "grid1": grid1,
    }


def train_step(
    model: LoopEarDSSL,
    optimizer: torch.optim.Optimizer,
    mel1: torch.Tensor,
    mel2: torch.Tensor,
    frame_valid1: torch.Tensor,
    frame_valid2: torch.Tensor,
    generator: torch.Generator,
    objective: str,
    step: int,
    total_steps: int,
) -> dict[str, float]:
    model.train()
    lr = lr_at(step, total_steps)
    set_lr(optimizer, lr)
    out = forward_batch(
        model, mel1, mel2, frame_valid1, frame_valid2, generator, objective
    )
    loss = out["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.student_parameters(), GRAD_CLIP)
    optimizer.step()
    momentum = ema_at(step, total_steps)
    model.ema_update(momentum)
    return {
        "loss": float(loss.detach()),
        "loss_g": float(out["loss_g"].detach()),
        "loss_l": float(out["loss_l"].detach()),
        "lr": float(lr),
        "ema": float(momentum),
        "grad_norm": float(grad_norm),
    }


def capture_rng() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(
    path: str,
    model: LoopEarDSSL,
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    objective: str,
    config: dict[str, Any],
) -> None:
    payload = {
        "step": int(step),
        "total_steps": int(total_steps),
        "objective": objective,
        "student_encoder": model.encoder.state_dict(),
        "student_predictor": model.predictor.state_dict(),
        "student_global": model.global_head.state_dict(),
        "student_local": model.local_head.state_dict(),
        "teacher_encoder": model.teacher_encoder.state_dict(),
        "teacher_local": model.teacher_local.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng(),
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: LoopEarDSSL,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(payload["student_encoder"])
    model.predictor.load_state_dict(payload["student_predictor"])
    model.global_head.load_state_dict(payload["student_global"])
    model.local_head.load_state_dict(payload["student_local"])
    model.teacher_encoder.load_state_dict(payload["teacher_encoder"])
    model.teacher_local.load_state_dict(payload["teacher_local"])
    for module in model.teacher_modules():
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    return payload


def init_model(seed: int) -> LoopEarDSSL:
    set_seed(seed)
    model = LoopEarDSSL()
    model.reset_teacher()
    return model


def snapshot_teacher(model: LoopEarDSSL) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for prefix, module in (
        ("encoder", model.teacher_encoder),
        ("local", model.teacher_local),
    ):
        for name, param in module.named_parameters():
            out[f"{prefix}.{name}"] = param.detach().clone()
    return out


def teacher_matches_formula(
    model: LoopEarDSSL, momentum: float, teacher_before: dict[str, torch.Tensor]
) -> bool:
    current_student = {
        "encoder": dict(model.encoder.named_parameters()),
        "local": dict(model.local_head.named_parameters()),
    }
    for key, old in teacher_before.items():
        prefix, name = key.split(".", 1)
        student = current_student[prefix][name].detach()
        expected = momentum * old + (1.0 - momentum) * student
        if not torch.allclose(model.teacher_encoder.state_dict()[name] if prefix == "encoder" else model.teacher_local.state_dict()[name], expected, atol=1e-6, rtol=1e-5):
            return False
    return True
