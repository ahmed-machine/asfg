"""
Optuna hyperparameter search for SSD fine-tuning.

Searches over lr, weight_decay, conf_weight (RoMa), and gamma (SEA-RAFT).
Uses SQLite storage for persistent trial history.

Usage:
    python3 scripts/experimental/ssd/tune.py -d data/ssd_labels --n-trials 20 --roma-only -e 1
    python3 scripts/experimental/ssd/tune.py -d data/ssd_labels --n-trials 20 --searaft-only -e 1
    python3 scripts/experimental/ssd/tune.py -d data/ssd_labels --n-trials 30 -e 2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

from align.models import get_torch_device
from scripts.experimental.ssd.finetune import train_roma_ssd, train_searaft_ssd


def make_roma_objective(data_dir, pairs_dir, device, epochs, study_dir):
    def objective(trial):
        lr = trial.suggest_float("roma_lr", 1e-6, 1e-4, log=True)
        weight_decay = trial.suggest_float("roma_weight_decay", 1e-5, 1e-2, log=True)
        conf_weight = trial.suggest_float("roma_conf_weight", 0.1, 2.0)

        run_dir = os.path.join(study_dir, f"trial_{trial.number}")

        print(f"\n--- RoMa Trial {trial.number} ---")
        print(f"  lr={lr:.2e} wd={weight_decay:.2e} conf_w={conf_weight:.3f}")

        try:
            loss = train_roma_ssd(
                data_dir, pairs_dir, device,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                conf_weight=conf_weight, run_dir=run_dir,
                log_every=50, use_tb=False, trial=trial,
            )
        except Exception as e:
            if "prune" in str(type(e).__name__).lower() or "Pruned" in str(e):
                raise optuna.TrialPruned()
            print(f"  Trial {trial.number} failed: {e}")
            return float('inf')

        if loss != loss:  # NaN check
            return float('inf')

        print(f"  Trial {trial.number}: final_loss={loss:.6f}")
        return loss

    return objective


def make_searaft_objective(data_dir, pairs_dir, device, epochs, study_dir):
    def objective(trial):
        lr = trial.suggest_float("searaft_lr", 5e-6, 5e-4, log=True)
        weight_decay = trial.suggest_float("searaft_weight_decay", 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float("searaft_gamma", 0.7, 0.95)

        run_dir = os.path.join(study_dir, f"trial_{trial.number}")

        print(f"\n--- SEA-RAFT Trial {trial.number} ---")
        print(f"  lr={lr:.2e} wd={weight_decay:.2e} gamma={gamma:.3f}")

        try:
            loss = train_searaft_ssd(
                data_dir, pairs_dir, device,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                gamma=gamma, run_dir=run_dir,
                log_every=50, use_tb=False, trial=trial,
            )
        except Exception as e:
            if "prune" in str(type(e).__name__).lower() or "Pruned" in str(e):
                raise optuna.TrialPruned()
            print(f"  Trial {trial.number} failed: {e}")
            return float('inf')

        if loss != loss:
            return float('inf')

        print(f"  Trial {trial.number}: final_loss={loss:.6f}")
        return loss

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna HP search for SSD fine-tuning")
    parser.add_argument("--data-dir", "-d", type=str, required=True)
    parser.add_argument("--pairs-dir", "-p", type=str, default=None)
    parser.add_argument("--epochs", "-e", type=int, default=1)
    parser.add_argument("--n-trials", "-n", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roma-only", action="store_true")
    parser.add_argument("--searaft-only", action="store_true")
    args = parser.parse_args()

    device = get_torch_device()
    print(f"Using device: {device}")

    pairs_dir = args.pairs_dir
    if pairs_dir is None:
        pairs_dir = args.data_dir.replace("ssd_labels", "ssd_pairs")
        if not os.path.isdir(pairs_dir):
            pairs_dir = os.path.join(os.path.dirname(args.data_dir), "ssd_pairs")

    if not os.path.isdir(pairs_dir):
        print(f"ERROR: Could not find pairs directory at {pairs_dir}")
        sys.exit(1)

    study_dir = os.path.join("diagnostics", "ssd_tune")
    os.makedirs(study_dir, exist_ok=True)
    db_path = os.path.join(study_dir, "ssd_tune.db")
    storage = f"sqlite:///{db_path}"

    t0 = time.time()

    # Run RoMa tuning
    if not args.searaft_only:
        study_name = "ssd_roma"
        print(f"\n{'='*60}")
        print(f"RoMa HP Search: {args.n_trials} trials, {args.epochs} epoch(s)")
        print(f"Storage: {db_path}")
        print(f"{'='*60}")

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=args.seed, n_startup_trials=5, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
            storage=storage,
            load_if_exists=True,
        )

        objective = make_roma_objective(
            args.data_dir, pairs_dir, device, args.epochs, study_dir)

        try:
            study.optimize(objective, n_trials=args.n_trials,
                           timeout=args.timeout, gc_after_trial=True)
        except KeyboardInterrupt:
            print("\nInterrupted.")

        _print_best(study, "roma", study_dir)

    # Run SEA-RAFT tuning
    if not args.roma_only:
        study_name = "ssd_searaft"
        print(f"\n{'='*60}")
        print(f"SEA-RAFT HP Search: {args.n_trials} trials, {args.epochs} epoch(s)")
        print(f"Storage: {db_path}")
        print(f"{'='*60}")

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=args.seed, n_startup_trials=5, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
            storage=storage,
            load_if_exists=True,
        )

        objective = make_searaft_objective(
            args.data_dir, pairs_dir, device, args.epochs, study_dir)

        try:
            study.optimize(objective, n_trials=args.n_trials,
                           timeout=args.timeout, gc_after_trial=True)
        except KeyboardInterrupt:
            print("\nInterrupted.")

        _print_best(study, "searaft", study_dir)

    total = time.time() - t0
    print(f"\nTotal time: {total:.0f}s")


def _print_best(study, model_name, study_dir):
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} — {len(study.trials)} trials completed")
    try:
        best = study.best_trial
        print(f"  Best loss: {best.value:.6f} (trial {best.number})")
        print(f"  Best params:")
        for k, v in sorted(best.params.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.6g}")
            else:
                print(f"    {k}: {v}")

        best_path = os.path.join(study_dir, f"best_{model_name}.json")
        with open(best_path, "w") as f:
            json.dump({
                "model": model_name,
                "score": best.value,
                "trial_number": best.number,
                "params": best.params,
                "total_trials": len(study.trials),
            }, f, indent=2)
        print(f"  Saved to {best_path}")
    except ValueError:
        print("  No completed trials.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
