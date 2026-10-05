"""Membership-inference audit of the toy over several seeds (python scripts/audit.py --seeds 0 1 2 3).

Each seed builds a fresh world, consolidates it, honours the demo's deletion requests and audits the erased facts
against facts of people never mentioned. Exact erasure should give AUCs scattered around 0.5, with intervals that
contain it; one adapter unlearned approximately (gradient ascent, NPO) should not.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slate.demo import run  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args(argv)
    print(f"{'seed':>4}  {'hash':>5}  {'SLATE AUC [95% CI]':>22}  {'before':>6}  {'GA':>5}  {'NPO':>5}")
    for seed in args.seeds:
        r = run(seed=seed, quiet=True)
        ci = f"{r['auc']:.2f} [{r['auc_ci'][0]:.2f}, {r['auc_ci'][1]:.2f}]"
        print(f"{seed:>4}  {str(r['hash_equal']):>5}  {ci:>22}  {r['auc_pre']:>6.2f}  "
              f"{r['approx']['gradient ascent']['auc']:>5.2f}  {r['approx']['NPO + retain']['auc']:>5.2f}")


if __name__ == "__main__":
    main()
