"""
scripts/fetch_iovnbd.py

Fetches the real IO-VNBD dataset (github.com/onyekpeu/IO-VNBD) into data/IO-VNBD-repo/.

The repo stores its CSVs in Git LFS: ~1 MB of git objects, ~3.7 GB of actual data. A plain
clone can hang partway through LFS filtering, so this retries the checkout rather than
leaving a tree full of unresolved pointer files - which the loader would otherwise read as
valid CSVs containing nonsense.

  python scripts/fetch_iovnbd.py            # everything (~3.7 GB, 564 CSVs)
  python scripts/fetch_iovnbd.py --subset   # synchronised V-/S- pairs only
  python scripts/fetch_iovnbd.py --verify   # check what is on disk, fetch nothing
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = "https://github.com/onyekpeu/IO-VNBD.git"
DEST = os.path.join(PROJECT_ROOT, "data", "IO-VNBD-repo")
SYNC_DIR = os.path.join(DEST, "Synchronised V abd S datasets", "Categorised IOVNB Dataset")

# A Git LFS pointer file is ~130 bytes of text. Any "CSV" smaller than this is not data.
LFS_POINTER_MAX_BYTES = 1024


def run(cmd, cwd=None, check=True):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r.returncode


def scan():
    """Return (real_csvs, pointer_csvs) under DEST."""
    real, pointers = [], []
    for root, _, files in os.walk(DEST):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            p = os.path.join(root, f)
            (pointers if os.path.getsize(p) <= LFS_POINTER_MAX_BYTES else real).append(p)
    return real, pointers


def verify(strict: bool = False) -> bool:
    if not os.path.isdir(DEST):
        print(f"[verify] {DEST} does not exist.")
        return False
    real, pointers = scan()
    print(f"[verify] {len(real)} real CSVs, {len(pointers)} unresolved LFS pointers")
    if pointers:
        print("         unresolved (first 5):")
        for p in pointers[:5]:
            print(f"           {os.path.relpath(p, PROJECT_ROOT)}")
        print("         re-run without --verify to finish the checkout.")
        return False
    if os.path.isdir(SYNC_DIR):
        drivers = sorted(os.listdir(SYNC_DIR))
        print(f"[verify] driver groups: {drivers}")
    return len(real) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", action="store_true",
                    help="fetch only the synchronised V-/S- pairs, skipping the "
                         "unsynchronised sets and the bundled .zip")
    ap.add_argument("--verify", action="store_true", help="report on-disk state, fetch nothing")
    ap.add_argument("--retries", type=int, default=4,
                    help="LFS checkout attempts (the remote hangs up on large batches)")
    args = ap.parse_args()

    if args.verify:
        sys.exit(0 if verify() else 1)

    os.makedirs(os.path.dirname(DEST), exist_ok=True)

    if not os.path.isdir(os.path.join(DEST, ".git")):
        print(f"[fetch] cloning {REPO_URL}")
        env = dict(os.environ)
        if args.subset:
            # Clone the git tree without pulling any LFS blob, then pull only what we want.
            env["GIT_LFS_SKIP_SMUDGE"] = "1"
            print("  (--subset: skipping LFS smudge, will pull selected paths after)")
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, DEST],
                           env=env, text=True, check=True)
        else:
            # A full clone often dies partway through filtering; the retry loop below
            # finishes whatever is left, so a non-zero exit here is not fatal.
            run(["git", "clone", "--depth", "1", REPO_URL, DEST], check=False)
    else:
        print(f"[fetch] {DEST} already cloned, resuming checkout")

    if args.subset:
        pattern = "Synchronised V abd S datasets/Categorised IOVNB Dataset/**"
        print(f"[fetch] pulling LFS objects for: {pattern}")
        run(["git", "lfs", "pull", "--include", pattern], cwd=DEST, check=False)
        run(["git", "checkout", "--", "Synchronised V abd S datasets"], cwd=DEST, check=False)

    for attempt in range(1, args.retries + 1):
        real, pointers = scan()
        if not pointers and real:
            break
        print(f"[fetch] attempt {attempt}/{args.retries}: "
              f"{len(real)} real, {len(pointers)} still pointers")
        run(["git", "restore", "--source=HEAD", ":/"], cwd=DEST, check=False)

    print()
    ok = verify()
    if ok:
        real, _ = scan()
        total_gb = sum(os.path.getsize(p) for p in real) / 1e9
        print(f"[fetch] done: {len(real)} CSVs, {total_gb:.2f} GB")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
