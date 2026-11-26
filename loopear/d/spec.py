import math

SR: int = 16000
N_MELS: int = 64
WIN_SAMPLES: int = 400
HOP_SAMPLES: int = 160
N_FFT: int = 400
N_FRAMES: int = 200
PATCH_FRAMES: int = 4
PATCH_BINS: int = 16
GRID_T: int = N_FRAMES // PATCH_FRAMES
GRID_F: int = N_MELS // PATCH_BINS
PATCH_DIM: int = PATCH_FRAMES * PATCH_BINS
D_MODEL: int = 96
N_LAYERS: int = 6
N_HEADS: int = 4
FFN_RATIO: int = 4
D_PRED: int = 48
PRED_LAYERS: int = 2
PRED_HEADS: int = 4
D_GLOBAL: int = 64
D_LOCAL: int = 64
GLOBAL_HIDDEN: int = 128
CROP_SAMPLES: int = WIN_SAMPLES + (N_FRAMES - 1) * HOP_SAMPLES
CROP_SECONDS: float = CROP_SAMPLES / float(SR)
TIME_MASK_RATIO: float = 0.60
FREQ_MASK_RATIO: float = 0.40
FREQ_MASK_BANDS: int = int(round(FREQ_MASK_RATIO * GRID_F))
TAU_GLOBAL: float = 0.1
TAU_LOCAL: float = 0.1
LAMBDA_LOCAL: float = 1.0
EMA_START: float = 0.99
EMA_END: float = 0.9995
LR: float = 3e-4
MIN_LR: float = 1e-6
WEIGHT_DECAY: float = 0.05
GRAD_CLIP: float = 1.0
SEED: int = 42
LOGMEL_EPS: float = 1e-6
H5_SECONDS: float = 5.0 * 3600.0
H20_SECONDS: float = 20.0 * 3600.0
REPACK_REPO: str = "quinnlue/FSD50K-16k"
REPACK_REVISION: str = "2a60d475f4e2f0db624a902881af2df79d656145"
UPSTREAM_REPO: str = "Fhrozen/FSD50k"
UPSTREAM_REVISION: str = "ccf1acaff12f3f4a4c10052dddafbb6d22152c9b"
ZENODO_DOI: str = "10.5281/zenodo.4060432"
K1_PRIOR: tuple[float, float, float] = (0.1, 0.8, 0.1)


def block_params(d_model: int, ffn_ratio: int) -> int:
    ln = 2 * (2 * d_model)
    attn = 4 * d_model * d_model + 4 * d_model
    hidden = ffn_ratio * d_model
    ffn = (d_model * hidden + hidden) + (hidden * d_model + d_model)
    return ln + attn + ffn


def encoder_param_count() -> int:
    blocks = N_LAYERS * block_params(D_MODEL, FFN_RATIO)
    final_ln = 2 * D_MODEL
    patch = PATCH_DIM * D_MODEL + D_MODEL
    cls = D_MODEL
    pos = (GRID_T + GRID_F) * D_MODEL
    return blocks + final_ln + patch + cls + pos


def predictor_param_count() -> int:
    proj_in = D_MODEL * D_PRED + D_PRED
    view = 2 * D_PRED
    blocks = PRED_LAYERS * block_params(D_PRED, FFN_RATIO)
    final_ln = 2 * D_PRED
    proj_out = D_PRED * D_MODEL + D_MODEL
    mask = D_MODEL
    return proj_in + view + blocks + final_ln + proj_out + mask


def global_head_param_count() -> int:
    ln = 2 * D_MODEL
    fc1 = D_MODEL * GLOBAL_HIDDEN + GLOBAL_HIDDEN
    fc2 = GLOBAL_HIDDEN * D_GLOBAL + D_GLOBAL
    return ln + fc1 + fc2


def local_head_param_count() -> int:
    ln = 2 * D_MODEL
    fc1 = D_MODEL * D_MODEL + D_MODEL
    fc2 = D_MODEL * D_LOCAL + D_LOCAL
    return ln + fc1 + fc2


ENCODER_PARAMS: int = encoder_param_count()
PREDICTOR_PARAMS: int = predictor_param_count()
GLOBAL_HEAD_PARAMS: int = global_head_param_count()
LOCAL_HEAD_PARAMS: int = local_head_param_count()
TRAIN_ONLY_PARAMS: int = PREDICTOR_PARAMS + GLOBAL_HEAD_PARAMS + LOCAL_HEAD_PARAMS
STUDENT_PARAMS: int = ENCODER_PARAMS + TRAIN_ONLY_PARAMS
TEACHER_PARAMS: int = ENCODER_PARAMS + LOCAL_HEAD_PARAMS


def warmup_steps(total_steps: int) -> int:
    return max(1, int(total_steps) // 20)


def lr_at(step: int, total_steps: int) -> float:
    warm = warmup_steps(total_steps)
    if step < warm:
        return LR * float(step + 1) / float(warm)
    span = max(total_steps - warm, 1)
    progress = min(max(float(step - warm) / float(span), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (LR - MIN_LR) * cosine


def ema_at(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return EMA_END
    progress = min(max(float(step) / float(total_steps - 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return EMA_START + (EMA_END - EMA_START) * cosine


DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "scale",
        "detail": "LoopEar-D is a 96-d 6-block encoder, not AudioFM's ViT-Base.",
    },
    {
        "id": "frontend",
        "detail": (
            "16 kHz mono, 64-bin HTK log-mel, 25 ms window, 10 ms hop, n_fft 400, "
            "uncentered STFT, natural log with floor 1e-6. Crop is 32240 samples "
            "(2.015 s) so the mel has exactly 200 frames."
        ),
    },
    {
        "id": "patches",
        "detail": "Non-overlapping 4-frame by 16-bin patches, grid 50 by 4, not 16 by 16.",
    },
    {
        "id": "positions",
        "detail": "Encoder positions are learned factorized time plus frequency embeddings, crop-relative.",
    },
    {
        "id": "predictor",
        "detail": "Training-only predictor has 2 pre-norm blocks at dim 48, not 4 blocks at 384.",
    },
    {
        "id": "heads",
        "detail": "Global MLP is 96-128-64 and local MLP is 96-96-64, then L2 norm.",
    },
    {
        "id": "temperature",
        "detail": "Both temperatures are 0.1. AudioFM uses 0.1 global and 0.2 local.",
    },
    {
        "id": "optim",
        "detail": "AdamW lr 3e-4, not 6e-4. Batch is profiled, not 6144. No drop-path.",
    },
    {
        "id": "augmentation",
        "detail": "Each view is an independent random crop. Short clips are left-aligned and padded.",
    },
    {
        "id": "masks",
        "detail": (
            "Time mask is 60 percent of valid time patches. Frequency mask is 2 of 4 bands "
            "because round(0.4 * 4) is 2, so 50 percent of frequency patches."
        ),
    },
    {
        "id": "padding",
        "detail": "Padded patches are excluded from attention, pooling, and contrastive denominators.",
    },
    {
        "id": "data",
        "detail": "Pretraining audio is the official FSD50K train split only, not AudioSet.",
    },
    {
        "id": "norm_stats",
        "detail": "Log-mel mean and std are fit on the downloaded training subset, not AudioSet constants.",
    },
    {
        "id": "context",
        "detail": "The encoder is bidirectional inside one 2.015 s window. It is not strictly causal.",
    },
    {
        "id": "kept",
        "detail": (
            "Kept from AudioFM: complementary time and frequency masks, symmetric NT-Xent "
            "with both views as negatives, EMA teacher on the encoder and local head, "
            "masked latent discrimination against same-coordinate cross-recording targets, "
            "and K=1 prior weights 0.1, 0.8, 0.1 renormalized on eligible offsets."
        ),
    },
)
