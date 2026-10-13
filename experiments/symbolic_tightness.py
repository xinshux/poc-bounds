"""Runs the symbolic tightness verification and writes the solution reports."""

import argparse
import os

from pocbounds.symbolic import GaussianSolver


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--equ", default=os.path.join("data", "equ_33.txt"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out-dir", default="solutions")
    args = ap.parse_args()
    GaussianSolver(equ_path=args.equ, k=args.k, out_dir=args.out_dir).run()


if __name__ == "__main__":
    main()
