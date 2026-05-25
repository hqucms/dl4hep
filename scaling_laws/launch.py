"""Generate (and optionally execute) weaver commands for the IsoFLOP scaling-law grid.

Reads flops.csv (from flop_profile.py), iterates over 4 budgets x 12 models,
drops cells where D is out of [10k, 1M] or steps < 30, and emits one weaver
invocation per cell.

By default prints the run table and writes commands to runs/commands.sh.
With --run, executes them sequentially.

Dataset setup
-------------
The dataset can be downloaded from:
  Train: https://hqu.web.cern.ch/datasets/JetClassMini/train_1M_parquet/
  Val:   https://hqu.web.cern.ch/datasets/JetClassMini/val_1M_parquet/

Set the DATADIR environment variable to the parent directory containing
train_1M_parquet/ and val_1M_parquet/ subdirectories, or pass --data-dir.
"""

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path

# --- Configuration ---

REPO_DIR = Path(__file__).parent
FLOPS_CSV = REPO_DIR / "flops.csv"
RUNS_DIR = REPO_DIR / "runs"

WEAVER = "weaver"
DATA_CONFIG = str(REPO_DIR / "data" / "JetClassMini_10cls.yaml")
NETWORK_CONFIG = str(REPO_DIR / "networks" / "example_ParticleTransformer.py")

# (group_name, train_file_basename, val_file_basename)
CLASSES = [
    ("HToBB",       "HToBB_000.parquet",       "HToBB_120.parquet"),
    ("HToCC",       "HToCC_000.parquet",       "HToCC_120.parquet"),
    ("HToGG",       "HToGG_000.parquet",       "HToGG_120.parquet"),
    ("HToWW2Q1L",   "HToWW2Q1L_000.parquet",   "HToWW2Q1L_120.parquet"),
    ("HToWW4Q",     "HToWW4Q_000.parquet",     "HToWW4Q_120.parquet"),
    ("TTBarLep",    "TTBarLep_000.parquet",    "TTBarLep_120.parquet"),
    ("TTBar",       "TTBar_000.parquet",       "TTBar_120.parquet"),
    ("WToQQ",       "WToQQ_000.parquet",       "WToQQ_120.parquet"),
    ("ZJetsToNuNu", "ZJetsToNuNu_000.parquet", "ZJetsToNuNu_120.parquet"),
    ("ZToQQ",       "ZToQQ_000.parquet",       "ZToQQ_120.parquet"),
]

# IsoFLOP budgets (in total training FLOPs, fwd+bwd). Geometric spacing ~sqrt(10).
BUDGETS = {
    "C1": 3.0e12,
    "C2": 1.0e13,
    "C3": 3.0e13,
    "C4": 1.0e14,
}

TRAIN_POOL_SIZE = 1_000_000   # jets in train_1M_parquet
D_MIN = 10_000
D_MAX = 1_000_000

BATCH_SIZE = 512
BATCH_SIZE_VAL = 1024
BASE_LR = 1e-3                 # at d_ref
D_REF = 128                    # reference width for LR scaling
WARMUP_FRACTION = 0.25         # passed to weaver as --warmup-steps (fraction < 1 => of total steps)
NUM_EPOCHS = 1                 # strict single-pass: one full sweep of the D-jet pool
MIN_TOTAL_STEPS = 30           # drop cells with fewer optimizer steps than this


def scaled_lr(d: int) -> float:
    """LR ~ 1/sqrt(d): keeps activation scale stable across widths (Xavier-style)."""
    return BASE_LR * (D_REF / d) ** 0.5


# --- Helpers ---

def format_argv(argv: list[str]) -> str:
    """Format an argv list into a readable backslash-continued shell command."""
    multi_value_flags = {"--data-train", "--data-val"}
    lines = [shlex.quote(argv[0])]
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") or tok == "-o" or tok == "-p":
            j = i + 1
            while j < len(argv) and not (
                argv[j].startswith("--") or argv[j] in ("-o", "-p")
            ):
                j += 1
            values = argv[i + 1:j]
            if tok in multi_value_flags and len(values) > 1:
                lines.append("  " + shlex.quote(tok) + " " + shlex.quote(values[0]))
                for v in values[1:]:
                    lines.append("    " + shlex.quote(v))
            else:
                parts = [shlex.quote(tok)] + [shlex.quote(v) for v in values]
                lines.append("  " + " ".join(parts))
            i = j
        else:
            lines.append("  " + shlex.quote(tok))
            i += 1
    return " \\\n".join(lines)


def load_flops():
    rows = {}
    with FLOPS_CSV.open() as f:
        for row in csv.DictReader(f):
            rows[row["model_id"]] = {
                "d": int(row["d"]),
                "L": int(row["L"]),
                "H": int(row["H"]),
                "params": int(row["params"]),
                "fwd_flops": int(row["fwd_flops_per_jet"]),
                "train_flops": int(row["train_flops_per_jet"]),
            }
    return rows


def data_groups(directory: str, idx: int) -> list[str]:
    """idx=1 -> train file (col 1 of CLASSES); idx=2 -> val file."""
    return [f"{name}:{directory}/{CLASSES[i][idx]}" for i, (name, *_) in enumerate(CLASSES)]


def schedule(D: int) -> tuple[int, int]:
    """Return (num_epochs, samples_per_epoch) for dataset size D."""
    total_steps = D // BATCH_SIZE
    if total_steps < MIN_TOTAL_STEPS:
        raise ValueError(f"D={D} -> {total_steps} steps, below minimum {MIN_TOTAL_STEPS}")
    steps_per_epoch = total_steps // NUM_EPOCHS
    samples_per_epoch = steps_per_epoch * BATCH_SIZE
    return NUM_EPOCHS, samples_per_epoch


def build_command(model_id: str, budget_id: str, m: dict, D: int,
                  train_dir: str, val_dir: str, gpu: str = "0") -> tuple[str, list[str]]:
    """Build (run_name, command_argv) for one cell."""
    run_name = f"{model_id}_{budget_id}"
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    num_epochs, samples_per_epoch = schedule(D)
    data_fraction = D / TRAIN_POOL_SIZE

    d, L, H = m["d"], m["L"], m["H"]
    pair = [d // 2, d // 2]
    lr = scaled_lr(d)

    argv = [
        WEAVER,
        "--data-config", DATA_CONFIG,
        "--network-config", NETWORK_CONFIG,
        "--data-train", *data_groups(train_dir, 1),
        "--data-val", *data_groups(val_dir, 2),
        "--data-fraction", f"{data_fraction:.6f}",
        "--data-fraction-val", "1.0",
        "--fetch-step", f"{data_fraction:.6f}",
        "--fetch-step-val", "1.0",
        "--in-memory",
        "--in-memory-val",
        "--num-workers", "1",
        "--batch-size", str(BATCH_SIZE),
        "--batch-size-val", str(BATCH_SIZE_VAL),
        "--num-epochs", str(num_epochs),
        "--samples-per-epoch", str(samples_per_epoch),
        "--samples-per-epoch-val", "1000000",
        "--optimizer", "adamW",
        "--start-lr", f"{lr:.6g}",
        "--lr-scheduler", "warmup+cos",
        "--warmup-steps", str(WARMUP_FRACTION),
        "--use-amp",
        "--amp-dtype", "bf16",
        # network kwargs
        "-o", "version", "3",
        "-o", "embed_dims", f"[{d}]",
        "-o", "pair_embed_dims", f"[{pair[0]},{pair[1]}]",
        "-o", "num_heads", str(H),
        "-o", "num_layers", str(L),
        "-o", "num_cls_layers", "0",
        "-o", "include_global_token", "True",
        "-o", "use_pre_activation_pair", "True",
        "-o", "block_params", '{"drop_path_rate":0.0}',
        # outputs
        "--model-prefix", str(out_dir / "net"),
        "--log", str(out_dir / "train.log"),
        "--tensorboard", f"scaling_{run_name}",
        "--gpus", gpu,
    ]
    return run_name, argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="Execute commands sequentially. Default: print + write commands.sh only.")
    ap.add_argument("--only", default=None,
                    help="Filter run_name (e.g. M3_C2). Default: all cells.")
    ap.add_argument("--cells", default=None,
                    help="Comma-separated subset of run_names to execute, e.g. M1_C1,M3_C2.")
    ap.add_argument("--gpu", default="0",
                    help="GPU index to pass to weaver via --gpus. Default 0.")
    ap.add_argument("--data-dir", default=None,
                    help="Parent directory containing train_1M_parquet/ and val_1M_parquet/.")
    ap.add_argument("--dry-run-weaver", action="store_true",
                    help="When --run, pass --print to weaver to dry-run model build only.")
    args = ap.parse_args()

    # Resolve data directories
    data_dir = args.data_dir or os.environ.get("DATADIR", "./JetClassMini")
    train_dir = os.path.join(data_dir, "train_1M_parquet")
    val_dir = os.path.join(data_dir, "val_1M_parquet")

    flops = load_flops()
    RUNS_DIR.mkdir(exist_ok=True)

    print(f"{'run':>8s}  {'model':>5s}  {'budget':>7s}  {'params':>8s}  "
          f"{'flops/jet':>12s}  {'D':>8s}  {'steps':>6s}  {'samples/ep':>10s}  "
          f"{'lr':>9s}  {'total_C':>9s}")
    print("-" * 102)

    cells = []
    for budget_id, C in BUDGETS.items():
        for model_id, m in flops.items():
            D = int(C / m["train_flops"])
            if D < D_MIN or D > D_MAX:
                continue
            total_steps = D // BATCH_SIZE
            if total_steps < MIN_TOTAL_STEPS:
                continue
            try:
                num_epochs, samples_per_epoch = schedule(D)
            except ValueError:
                continue
            actual_D = num_epochs * samples_per_epoch
            actual_C = actual_D * m["train_flops"]
            run_name = f"{model_id}_{budget_id}"
            cells.append((run_name, model_id, budget_id, m, D))
            print(f"{run_name:>8s}  {model_id:>5s}  {budget_id:>7s}  "
                  f"{m['params']:>8,d}  {m['train_flops']:>12,d}  "
                  f"{D:>8,d}  {total_steps:>6d}  {samples_per_epoch:>10,d}  "
                  f"{scaled_lr(m['d']):>9.2e}  {actual_C:>9.2e}")

    print(f"\nTotal cells: {len(cells)}")

    if args.only:
        cells = [c for c in cells if c[0] == args.only]
        print(f"After --only filter: {len(cells)} cell(s)")
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c[0] in wanted]
        print(f"After --cells filter: {len(cells)} cell(s)")

    # Write commands.sh
    cmds_path = RUNS_DIR / "commands.sh"
    with cmds_path.open("w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for run_name, model_id, budget_id, m, D in cells:
            _, argv = build_command(model_id, budget_id, m, D,
                                    train_dir, val_dir, gpu=args.gpu)
            f.write(f"# {run_name}: params={m['params']:,d} D={D:,d}\n")
            f.write(format_argv(argv))
            f.write("\n\n")
    os.chmod(cmds_path, 0o755)
    print(f"Wrote {cmds_path}")

    if not args.run:
        return

    # Execute sequentially
    print(f"\nExecuting {len(cells)} runs sequentially...")
    for i, (run_name, model_id, budget_id, m, D) in enumerate(cells, 1):
        _, argv = build_command(model_id, budget_id, m, D,
                                train_dir, val_dir, gpu=args.gpu)
        if args.dry_run_weaver:
            argv.append("--print")
        print(f"\n[{i}/{len(cells)}] === {run_name} (params={m['params']:,d} D={D:,d}) ===")
        result = subprocess.run(argv)
        if result.returncode != 0:
            print(f"!!! {run_name} failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
