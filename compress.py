"""Buat artefak varian terkompresi per fold: bobot pruning dan int8.onnx (fp16 tidak butuh artefak)."""
import argparse
import json
import shutil
import traceback
from pathlib import Path

from train import done_matches, drop_checkpoints, resolve_out, write_done

try:
    from onnxruntime.quantization import CalibrationDataReader as _ReaderBase
except ImportError:  # onnxruntime tidak terpasang: hanya int8 yang butuh
    _ReaderBase = object

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
    # hanya kunci yang dibutuhkan pemuat: salinan tensor lama (state_dict/callbacks) menggandakan ukuran file
    ckpt = {k: ckpt[k] for k in ("model", "args", "model_config") if k in ckpt}
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
    drop_checkpoints(run_dir)
    return {"kept_ffn_fraction": kept["fraction"]}


def build_prune_structured(fold: Path, dest: Path, args) -> dict:
    return prune_structured(fold, dest / "weights.pth", args.structured_ratio, args.finetune_epochs)


class ImageCalibrationReader(_ReaderBase):
    """Pembaca kalibrasi ORT: satu gambar per batch, diproses persis seperti RFDETR.predict()."""

    def __init__(self, paths, input_name: str, height: int, width: int):
        self.paths, self.input_name, self.height, self.width = list(paths), input_name, height, width
        self.rewind()

    def rewind(self) -> None:
        self._it = iter(self.paths)

    def get_next(self):
        from PIL import Image
        from rfdetr.export._runtime.preprocess import preprocess_to_nchw  # modul privat; rfdetr dipin ke 1.11.2

        path = next(self._it, None)
        if path is None:
            return None
        with Image.open(path) as img:
            return {self.input_name: preprocess_to_nchw(img, self.height, self.width, 3)}


def export_int8(weights: Path, num_classes: int, calib_images: list, out_path: Path, n_calib: int = 100) -> Path:
    """Ekspor ONNX (fp32) lalu kuantisasi statis INT8 (QDQ, per-channel) dengan kalibrasi dari calib_images."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from rfdetr import RFDETRSegSmall

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = RFDETRSegSmall(pretrain_weights=str(weights), num_classes=num_classes)
    fp32 = Path(model.export(output_dir=str(out_path.parent / "onnx_fp32"), format="onnx", fp16=False, verbose=False))
    inp = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).get_inputs()[0]
    _, _, h, w = inp.shape
    reader = ImageCalibrationReader(calib_images[:n_calib], inp.name, h, w)
    quantize_static(
        str(fp32), str(out_path), reader, quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
    )
    return out_path


def build_int8(fold: Path, dest: Path, args) -> dict:
    # kalibrasi hanya dari fold latih (bukan fold uji): folder dataset/train fold ini
    imgs = sorted((fold / "dataset" / "train").glob("*.jpg")) + sorted((fold / "dataset" / "train").glob("*.png"))
    num_classes = len(json.loads((fold / "classes.json").read_text(encoding="utf-8")))
    out = export_int8(fold / "weights.pth", num_classes, imgs, dest / "model_int8.onnx", args.n_calib)
    return {"onnx": out.name, "n_calib": min(args.n_calib, len(imgs))}


def build_prune_unstructured(fold: Path, dest: Path, args) -> dict:
    return {"sparsity": prune_unstructured(fold / "weights.pth", dest / "weights.pth", args.prune_amount)}


BUILDERS = {"int8": build_int8, "prune_unstructured": build_prune_unstructured, "prune_structured": build_prune_structured}


VARIANT_SETTINGS = {
    "int8": lambda a: {"n_calib": a.n_calib},
    "prune_unstructured": lambda a: {"prune_amount": a.prune_amount},
    "prune_structured": lambda a: {"structured_ratio": a.structured_ratio, "finetune_epochs": a.finetune_epochs},
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--prune-amount", type=float, default=0.5, help="sparsitas prune_unstructured")
    ap.add_argument("--structured-ratio", type=float, default=0.3, help="fraksi neuron FFN yang dibuang")
    ap.add_argument("--finetune-epochs", type=int, default=5)
    ap.add_argument("--n-calib", type=int, default=100, help="gambar kalibrasi INT8")
    ap.add_argument("--smoke", action="store_true", help="1 fold")
    ap.add_argument("--out", type=Path, default=None, help="default outputs (outputs_smoke untuk --smoke)")
    a = ap.parse_args(argv)
    if a.smoke:
        a.folds = 1
    a.out = resolve_out(a.out, a.smoke)
    for k in range(a.folds):
        fold = a.out / f"fold{k}"
        if not (fold / "DONE").exists():
            raise SystemExit(f"{fold}/DONE tidak ada: jalankan train.py dulu.")
        for v in a.variants:
            dest = fold / "variants" / v
            # stempel: pengaturan varian + penanda fold (jika fold dilatih ulang, varian dibuat ulang)
            settings = {"fold": (fold / "DONE").read_text(encoding="utf-8"), **VARIANT_SETTINGS[v](a)}
            if done_matches(dest / "DONE", settings):
                print(f"fold {k} {v}: sudah ada, dilewati")
                continue
            (dest / "DONE").unlink(missing_ok=True)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "FAILED.txt").unlink(missing_ok=True)
            try:
                info = BUILDERS[v](fold, dest, a)
                (dest / "info.json").write_text(json.dumps(info), encoding="utf-8")
                write_done(dest / "DONE", settings)
                print(f"fold {k} {v}: selesai {info}")
            except Exception as e:  # noqa: BLE001 - varian gagal ditandai, varian lain tetap jalan
                (dest / "FAILED.txt").write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}", encoding="utf-8")
                print(f"fold {k} {v}: GAGAL ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
