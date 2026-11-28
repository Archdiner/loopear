import torch
import torch.nn.functional as F


def embedding_diagnostics(z: torch.Tensor) -> dict[str, float | int | bool]:
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError(f"diagnostics need [N, D] with N>=2, got {tuple(z.shape)}")
    if not torch.isfinite(z).all():
        return {
            "n": int(z.shape[0]),
            "dim": int(z.shape[1]),
            "variance_mean": float("nan"),
            "variance_min": float("nan"),
            "effective_rank": float("nan"),
            "offdiag_cosine_mean": float("nan"),
            "singular_max": float("nan"),
            "singular_min": float("nan"),
            "collapsed": True,
            "finite": False,
        }
    var = z.detach().float().var(dim=0, unbiased=False)
    centered = z.detach().float() - z.detach().float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum().clamp_min(1e-12)
    prob = singular / total
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum()
    eff_rank = torch.exp(entropy)
    zn = F.normalize(z.detach().float(), dim=-1, eps=1e-6)
    sim = zn @ zn.T
    n = int(z.shape[0])
    off = (sim.sum() - sim.diag().sum()) / float(n * (n - 1))
    collapsed = bool((float(off) > 0.95 and float(eff_rank) < 2.0) or float(var.mean()) < 1e-8)
    return {
        "n": n,
        "dim": int(z.shape[1]),
        "variance_mean": float(var.mean()),
        "variance_min": float(var.min()),
        "effective_rank": float(eff_rank),
        "offdiag_cosine_mean": float(off),
        "singular_max": float(singular[0]),
        "singular_min": float(singular[-1]),
        "collapsed": collapsed,
        "finite": True,
    }


def param_delta_l2(current: torch.Tensor, initial: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(current.detach().float() - initial.detach().float()))


def flat_params(module: torch.nn.Module) -> torch.Tensor:
    parts = [p.detach().float().reshape(-1) for p in module.parameters()]
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def cpu_seconds() -> float:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def peak_rss_gb() -> float:
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def analytic_encoder_flops(batch: int = 1, tokens_with_cls: int = 201) -> dict[str, float]:
    d = 96
    layers = 6
    patch_macs = batch * 200 * 64 * d
    linear_macs = batch * layers * tokens_with_cls * (4 * d * d + 8 * d * d)
    attn_macs = batch * layers * (2 * tokens_with_cls * tokens_with_cls * d)
    macs = patch_macs + linear_macs + attn_macs
    return {
        "batch": float(batch),
        "tokens_with_cls": float(tokens_with_cls),
        "gemm_macs": float(macs),
        "gemm_flops_mul_add": float(2 * macs),
        "gemm_gflops": float(2 * macs) / 1e9,
    }


def hours(seconds: float) -> float:
    return float(seconds) / 3600.0
