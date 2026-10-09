"""Buat artefak varian terkompresi per fold: bobot pruning dan int8.onnx (fp16 tidak butuh artefak)."""
import argparse
import json
import shutil
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


FFN_NAMES = (("linear1", "linear2"), ("fc1", "fc2"))  # decoder RF-DETR dan MLP backbone DINOv2


def prune_ffn_pair(l1, l2, ratio: float):
    """Buang neuron FFN terkecil (norma L1 baris l1 + kolom l2); kembalikan (Linear baru, Linear baru) yang lebih kecil."""
    import torch.nn as nn

    keep = max(1, round(l1.out_features * (1 - ratio)))
    imp = l1.weight.abs().sum(1) + l2.weight.abs().sum(0)
    idx = imp.topk(keep).indices.sort().values
    n1 = nn.Linear(l1.in_features, keep, bias=l1.bias is not None).to(l1.weight)
    n2 = nn.Linear(keep, l2.out_features, bias=l2.bias is not None).to(l2.weight)
    n1.weight.data.copy_(l1.weight.data[idx])
    n2.weight.data.copy_(l2.weight.data[:, idx])
    if l1.bias is not None:
        n1.bias.data.copy_(l1.bias.data[idx])
    if l2.bias is not None:
        n2.bias.data.copy_(l2.bias.data)
    return n1, n2


def _ffn_pairs(module):
    import torch.nn as nn

    for name, sub in module.named_modules():
        for a, b in FFN_NAMES:
            if isinstance(getattr(sub, a, None), nn.Linear) and isinstance(getattr(sub, b, None), nn.Linear):
                yield name, sub, a, b


def prune_ffn(module, ratio: float) -> float:
    """Pangkas semua pasangan FFN in-place (ganti layer); kembalikan fraksi neuron FFN yang dipertahankan."""
    total = kept = 0
    for _, owner, a, b in list(_ffn_pairs(module)):
        l1, l2 = getattr(owner, a), getattr(owner, b)
        n1, n2 = prune_ffn_pair(l1, l2, ratio)
        setattr(owner, a, n1)
        setattr(owner, b, n2)
        total += l1.out_features
        kept += n1.out_features
    return kept / total


def shrink_like(module, sd: dict) -> None:
    """Ubah dimensi FFN module agar cocok dengan state dict yang sudah dipangkas (sebelum load_state_dict)."""
    import torch.nn as nn

    for name, owner, a, b in list(_ffn_pairs(module)):
        l1, l2 = getattr(owner, a), getattr(owner, b)
        k = sd[f"{name}.{a}.weight" if name else f"{a}.weight"].shape[0]
        setattr(owner, a, nn.Linear(l1.in_features, k, bias=l1.bias is not None).to(l1.weight))
        setattr(owner, b, nn.Linear(k, l2.out_features, bias=l2.bias is not None).to(l2.weight))


def prune_structured(fold: Path, weights_out: Path, ratio: float = 0.3, finetune_epochs: int = 5) -> dict:
    """Pangkas FFN lalu fine-tune singkat di data latih fold. Bobot EMA akhir disimpan ke weights_out."""
    import rfdetr.training as T
    from rfdetr import RFDETRSegSmall

    num_classes = len(json.loads((fold / "classes.json").read_text(encoding="utf-8")))
    run_dir = weights_out.parent / "run"
    kept = {}
    orig = T.RFDETRModelModule

    class PrunedModule(orig):  # train() mengimpor kelas ini saat dipanggil; pangkas setelah bobot fold dimuat
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            kept["fraction"] = prune_ffn(self.model, ratio)

    T.RFDETRModelModule = PrunedModule
    try:
        RFDETRSegSmall(pretrain_weights=str(fold / "weights.pth"), num_classes=num_classes).train(
            dataset_dir=str(fold / "dataset"), epochs=finetune_epochs, batch_size="auto", lr=1e-4, output_dir=str(run_dir)
        )
    finally:
        T.RFDETRModelModule = orig
    shutil.copy2(run_dir / "last_ema.pth", weights_out)
    return {"kept_ffn_fraction": kept["fraction"]}


def build_prune_structured(fold: Path, dest: Path, args) -> dict:
    return prune_structured(fold, dest / "weights.pth", args.structured_ratio, args.finetune_epochs)


def build_prune_unstructured(fold: Path, dest: Path, args) -> dict:
    return {"sparsity": prune_unstructured(fold / "weights.pth", dest / "weights.pth", args.prune_amount)}


BUILDERS = {"prune_unstructured": build_prune_unstructured, "prune_structured": build_prune_structured}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--prune-amount", type=float, default=0.5, help="sparsitas prune_unstructured")
    ap.add_argument("--structured-ratio", type=float, default=0.3, help="fraksi neuron FFN yang dibuang")
    ap.add_argument("--finetune-epochs", type=int, default=5)
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
