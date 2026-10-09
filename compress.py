"""Buat artefak varian terkompresi per fold: bobot pruning dan int8.onnx (fp16 tidak butuh artefak)."""
import argparse
import json
import traceback
from pathlib import Path

PRUNE_SKIP = ("embed", "query_feat")  # embedding/query yang dipelajari tidak dipangkas
# ponytail: heuristik berdasarkan nama kunci (Linear/Conv/in_proj = tensor ndim>=2 ber-"weight"); ganti per-modul bila ada layer aneh


def prune_state_dict(sd: dict, amount: float):
    """Pruning L1 global pada bobot Linear/Conv. Mengembalikan (state dict baru, sparsitas aktual); input tidak dimutasi."""
    import torch

    keys = [k for k, v in sd.items() if torch.is_tensor(v) and v.ndim >= 2 and "weight" in k and not any(s in k for s in PRUNE_SKIP)]
    mags = torch.cat([sd[k].abs().flatten().float() for k in keys])
    n = max(1, int(amount * mags.numel()))
    thr = mags.kthvalue(n).values
    out = dict(sd)
    zeros = 0
    for k in keys:
        out[k] = torch.where(sd[k].abs() <= thr, torch.zeros_like(sd[k]), sd[k])
        zeros += int((out[k] == 0).sum())
    return out, zeros / mags.numel()


def prune_unstructured(weights_in: Path, weights_out: Path, amount: float = 0.5) -> float:
    """Pangkas weights.pth (bobot dinolkan) dan simpan dengan format checkpoint yang sama; kembalikan sparsitas."""
    import torch

    ckpt = torch.load(weights_in, map_location="cpu", weights_only=False)  # berkas buatan kita sendiri
    ckpt["model"], sparsity = prune_state_dict(ckpt["model"], amount)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, weights_out)
    return sparsity


def build_prune_unstructured(fold: Path, dest: Path, args) -> dict:
    return {"sparsity": prune_unstructured(fold / "weights.pth", dest / "weights.pth", args.prune_amount)}


BUILDERS = {"prune_unstructured": build_prune_unstructured}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--prune-amount", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true", help="1 fold")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    a = ap.parse_args(argv)
    if a.smoke:
        a.folds = 1
    for k in range(a.folds):
        fold = a.out / f"fold{k}"
        if not (fold / "DONE").exists():
            raise SystemExit(f"{fold}/DONE tidak ada: jalankan train.py dulu.")
        for v in a.variants:
            dest = fold / "variants" / v
            if (dest / "DONE").exists():
                print(f"fold {k} {v}: sudah ada, dilewati")
                continue
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "FAILED.txt").unlink(missing_ok=True)
            try:
                info = BUILDERS[v](fold, dest, a)
                (dest / "info.json").write_text(json.dumps(info), encoding="utf-8")
                (dest / "DONE").write_text("ok")
                print(f"fold {k} {v}: selesai {info}")
            except Exception as e:  # noqa: BLE001 - varian gagal ditandai, varian lain tetap jalan
                (dest / "FAILED.txt").write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}", encoding="utf-8")
                print(f"fold {k} {v}: GAGAL ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
