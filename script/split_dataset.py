# -*- coding: utf-8 -*-
# Usage:
#   python -m script.split_debug --src data/asl_alphabet_train --out data/split --cap 500 --seed 42
import argparse, random, shutil
from pathlib import Path

WHITELIST = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space", "nothing"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def pick_files(folder: Path, cap: int, seed: int):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    rnd = random.Random(seed); rnd.shuffle(files)
    if cap is None or cap <= 0:  # <- allow no cap
        return files
    return files[:cap]

def split_three(n: int, p_train=0.7, p_val=0.15):
    n_train = int(round(n * p_train))
    n_val = int(round(n * p_val))
    n_test = n - n_train - n_val
    # Ajustement si arrondis posent problème
    while n_test < 0:
        n_val -= 1; n_test += 1
    return n_train, n_val, n_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="data/asl_alphabet_train")
    ap.add_argument("--out", type=Path, required=True, help="data/split")
    ap.add_argument("--cap", type=int, default=500,
                help="max images per class (global before split). Use -1 for no cap.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    args = ap.parse_args()

    src_root: Path = args.src
    out_root: Path = args.out

    # Crée arborescence
    for split in ("train", "val", "test"):
        for cls in WHITELIST:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    # Boucle classes
    for cls in WHITELIST:
        src_dir = src_root / cls
        if not src_dir.exists():
            print(f"[WARN] Classe absente: {cls} ({src_dir})")
            continue

        # Tirage aléatoire reproductible
        files = pick_files(src_dir, args.cap, seed=args.seed)
        n = len(files)
        n_tr, n_va, n_te = split_three(n, p_train=args.train, p_val=args.val)

        # Répartition
        rnd = random.Random(args.seed)  # reshuffle pour distribution
        rnd.shuffle(files)
        train_files = files[:n_tr]
        val_files = files[n_tr:n_tr+n_va]
        test_files = files[n_tr+n_va:]

        # Copie
        for f in train_files:
            shutil.copy2(f, out_root / "train" / cls / f.name)
        for f in val_files:
            shutil.copy2(f, out_root / "val" / cls / f.name)
        for f in test_files:
            shutil.copy2(f, out_root / "test" / cls / f.name)

        print(f"{cls}: total={n} -> train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    print(f"Done -> {out_root}")

if __name__ == "__main__":
    main()
