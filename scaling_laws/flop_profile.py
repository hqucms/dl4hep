"""Profile per-jet FLOPs and param counts for the ParT model grid.

Profiles 12 model configurations (S1-S5, M1-M7) of varying width (d),
depth (L), and number of attention heads (H).

Run on CPU (fvcore traces the computation graph, doesn't time it).
Output: flops.csv with columns
    model_id, d, L, H, params, fwd_flops_per_jet, train_flops_per_jet
where train_flops_per_jet = 3 * fwd_flops_per_jet (standard estimate:
1x forward + 2x backward).

Requirements:
    pip install torch fvcore weaver-core
"""

import csv
import warnings
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis
from weaver.nn.model.ParticleTransformer import ParticleTransformer

warnings.filterwarnings("ignore")

INPUT_DIM = 17        # pf_features (17 features in JetClassMini_10cls.yaml)
SEQ_LEN = 128         # max particles per jet
NUM_CLASSES = 10

# Model grid: (id, embed_dim, num_layers, num_heads)
MODELS = [
    ("S1", 32,  1, 2),
    ("S2", 32,  2, 2),
    ("S3", 48,  1, 3),
    ("S4", 48,  2, 3),
    ("S5", 64,  1, 4),
    ("M1", 64,  2, 4),
    ("M2", 64,  4, 4),
    ("M3", 96,  2, 6),
    ("M4", 64,  6, 4),
    ("M5", 128, 2, 8),
    ("M6", 96,  4, 6),
    ("M7", 96,  6, 6),
]


def build_model(d: int, L: int, H: int) -> torch.nn.Module:
    return ParticleTransformer(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        # network
        pair_input_type="pp",
        pair_input_dim=4,
        use_pre_activation_pair=True,
        embed_dims=[d],
        pair_embed_dims=[d // 2, d // 2],
        num_heads=H,
        num_layers=L,
        block_params={"drop_path_rate": 0.0},
        include_global_token=True,
        num_cls_layers=0,
        fc_params=[],
        # misc
        version=3,
        trim=False,         # disable trimming so FLOPs are reproducible at seq_len=128
        for_inference=False,
    )


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_fwd_flops(model: torch.nn.Module, batch_size: int = 1) -> int:
    x = torch.randn(batch_size, INPUT_DIM, SEQ_LEN)
    v = torch.randn(batch_size, 4, SEQ_LEN)
    mask = torch.ones(batch_size, 1, SEQ_LEN, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, (x, v, mask))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
    return total // batch_size


def main():
    out_path = Path(__file__).parent / "flops.csv"
    rows = []
    for mid, d, L, H in MODELS:
        model = build_model(d, L, H)
        n_params = count_params(model)
        fwd = count_fwd_flops(model)
        train = 3 * fwd
        print(f"{mid}: d={d} L={L} H={H} "
              f"params={n_params:>9,d} "
              f"fwd_flops={fwd:>12,d} "
              f"train_flops={train:>12,d}")
        rows.append({
            "model_id": mid,
            "d": d,
            "L": L,
            "H": H,
            "params": n_params,
            "fwd_flops_per_jet": fwd,
            "train_flops_per_jet": train,
        })

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
