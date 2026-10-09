"""Evaluasi semua varian: metrik, confusion matrix, biaya komputasi, visualisasi, summary.csv.

Impor berat (torch, rfdetr, onnxruntime) dilakukan di dalam fungsi yang memakainya agar fungsi murni bisa diuji tanpa GPU.
"""
import argparse
import contextlib
import csv
import io
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Instances:
    masks: np.ndarray  # (N, H, W) bool
    class_id: np.ndarray  # (N,) int, indeks kelas 0..C-1
    score: np.ndarray  # (N,) float


def match_instances(gt: Instances, pred: Instances, iou_thr: float = 0.5, score_thr: float = 0.5):
    """Pasangkan prediksi ke GT (greedy dari skor tertinggi, IoU mask, tiap GT sekali, tanpa melihat kelas).

    Mengembalikan daftar (gt_idx, pred_idx); None untuk GT/prediksi yang tidak berpasangan.
    Prediksi di bawah score_thr diabaikan sepenuhnya."""
    order = [j for j in np.argsort(-pred.score, kind="stable") if pred.score[j] >= score_thr]
    n_gt = len(gt.class_id)
    if n_gt and order:
        g = gt.masks.reshape(n_gt, -1).astype(np.float64)
        p = pred.masks.reshape(len(pred.class_id), -1).astype(np.float64)
        inter = p @ g.T
        union = p.sum(1)[:, None] + g.sum(1)[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    pairs, taken = [], set()
    for j in order:
        best, best_iou = None, iou_thr
        if n_gt:
            for i in np.argsort(-iou[j], kind="stable"):
                if int(i) not in taken and iou[j, i] >= best_iou:
                    best, best_iou = int(i), iou[j, i]
                    break
        if best is None:
            pairs.append((None, int(j)))
        else:
            taken.add(best)
            pairs.append((best, int(j)))
    pairs += [(i, None) for i in range(n_gt) if i not in taken]
    return pairs


def confusion_matrix(pairs, gt: Instances, pred: Instances, num_classes: int) -> np.ndarray:
    """(C+1, C+1): baris = kelas GT, kolom = kelas prediksi, indeks C = background."""
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    for gi, pj in pairs:
        r = num_classes if gi is None else int(gt.class_id[gi])
        c = num_classes if pj is None else int(pred.class_id[pj])
        cm[r, c] += 1
    return cm


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def summarize(cm: np.ndarray) -> dict:
    """Precision/recall/F1 (micro dan macro), accuracy deteksi dan klasifikasi dari confusion matrix.

    Pasangan cocok berkelas salah dihitung FP untuk kelas prediksi dan FN untuk kelas GT. 0/0 -> nan."""
    C = cm.shape[0] - 1
    tp = np.array([cm[c, c] for c in range(C)], float)
    fp = np.array([cm[:, c].sum() - cm[c, c] for c in range(C)], float)
    fn = np.array([cm[c, :].sum() - cm[c, c] for c in range(C)], float)
    per_class = {
        c: {"precision": _ratio(tp[c], tp[c] + fp[c]), "recall": _ratio(tp[c], tp[c] + fn[c]), "f1": _ratio(2 * tp[c], 2 * tp[c] + fp[c] + fn[c])}
        for c in range(C)
    }

    def macro(key):
        vals = [v[key] for v in per_class.values()]
        return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")

    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    matched = cm[:C, :C].sum()
    return {
        "precision_micro": _ratio(TP, TP + FP),
        "recall_micro": _ratio(TP, TP + FN),
        "f1_micro": _ratio(2 * TP, 2 * TP + FP + FN),
        "precision_macro": macro("precision"),
        "recall_macro": macro("recall"),
        "f1_macro": macro("f1"),
        "accuracy_detection": _ratio(TP, TP + FP + FN),
        "accuracy_classification": _ratio(TP, matched),
        "per_class": per_class,
    }


def coco_map(gt_json: Path, preds: list[dict], img_ids: list[int]) -> dict:
    """mAP50-95, mAP50, dan AP per kategori (COCOeval segm). preds berformat COCO (RLE). Kosong -> 0.0."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
        annotated = sorted({a["category_id"] for a in gt.dataset["annotations"]})
        if not preds:
            return {"map50_95": 0.0, "map50": 0.0, "ap_per_class": {c: 0.0 for c in annotated}}
        ev = COCOeval(gt, gt.loadRes(preds), "segm")
        ev.params.imgIds = list(img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    prec = ev.eval["precision"]  # (T, R, K, A, M); -1 = tanpa GT
    ap = {}
    for k, cat in enumerate(ev.params.catIds):
        v = prec[:, :, k, 0, -1]
        v = v[v > -1]
        ap[cat] = float(v.mean()) if v.size else float("nan")
    nan_if_neg = lambda x: float(x) if x >= 0 else float("nan")  # noqa: E731
    return {"map50_95": nan_if_neg(ev.stats[0]), "map50": nan_if_neg(ev.stats[1]), "ap_per_class": ap}


class Predictor(Protocol):
    name: str

    def predict(self, image, threshold: float):  # PIL.Image -> supervision.Detections (mask berukuran gambar asli)
        ...


class TorchPredictor:
    def __init__(self, name: str, model):
        self.name, self.model = name, model

    def predict(self, image, threshold: float):
        return self.model.predict(image, threshold=threshold)


def decode_masks(mask_logits: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Logit mask (K, Hm, Wm) -> bool (K, H, W): resize bilinear lalu > 0, sama seperti RFDETR.predict()."""
    import torch
    import torch.nn.functional as F

    k = mask_logits.shape[0]
    if k == 0:
        return np.zeros((0, *size), bool)
    t = torch.from_numpy(np.ascontiguousarray(mask_logits)).float().unsqueeze(1)
    out = [F.interpolate(t[i : i + 32], size=size, mode="bilinear", align_corners=False) > 0 for i in range(0, k, 32)]
    return torch.cat(out).squeeze(1).numpy()


class OrtPredictor:
    """Prediktor ONNX Runtime (varian int8): preprocess dan decode mengikuti referensi rfdetr; mask lewat decode_masks."""

    def __init__(self, name: str, onnx_path: Path):
        import onnxruntime as ort

        want = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider") if p in ort.get_available_providers()]
        want = [p for p in want if p != "TensorrtExecutionProvider"]  # TensorRT EP membangun engine saat pertama jalan; pakai CUDA EP
        self.name = name
        self.session = ort.InferenceSession(str(onnx_path), providers=want + ["CPUExecutionProvider"])
        self.providers = self.session.get_providers()  # dicatat ke summary: bila hanya CPU, latency tidak sebanding dengan GPU
        inp = self.session.get_inputs()[0]
        self.input_name, (_, _, self.h, self.w) = inp.name, inp.shape
        self.out_names = [o.name for o in self.session.get_outputs()]

    def predict(self, image, threshold: float):
        import supervision as sv
        from rfdetr.export._runtime.decode import decode_detections
        from rfdetr.export._runtime.preprocess import preprocess_to_nchw

        outs = dict(zip(self.out_names, self.session.run(None, {self.input_name: preprocess_to_nchw(image, self.h, self.w, 3)})))
        dec = decode_detections(outs["dets"][0], outs["labels"][0], image.size, threshold=threshold, background_class_id=-1)
        masks = decode_masks(outs["masks"][0][dec.query_index], (image.height, image.width))
        return sv.Detections(xyxy=dec.xyxy, confidence=dec.confidence, class_id=dec.class_id.astype(int), mask=masks)


def to_instances(det, size: tuple[int, int], num_classes: int | None = None) -> Instances:
    """sv.Detections -> Instances. size = (tinggi, lebar) gambar asli. Tanpa mask -> tidak ada instance.
    class_id >= num_classes adalah slot no-object rfdetr (muncul pada threshold rendah) dan dibuang."""
    if det.mask is None or len(det) == 0:
        return Instances(np.zeros((0, *size), bool), np.zeros(0, int), np.zeros(0, float))
    keep = np.ones(len(det), bool) if num_classes is None else det.class_id < num_classes
    return Instances(det.mask[keep].astype(bool), det.class_id[keep].astype(int), det.confidence[keep].astype(float))


def load_predictor(variant: str, fold_dir: Path, num_classes: int) -> Predictor:
    """Muat prediktor satu varian dari artefak fold. Varian lain ditambahkan di task berikutnya."""
    if variant in ("fp32", "fp16", "prune_unstructured"):
        from rfdetr import RFDETRSegSmall

        weights = Path(fold_dir) / ("variants/prune_unstructured/weights.pth" if variant == "prune_unstructured" else "weights.pth")
        model = RFDETRSegSmall(pretrain_weights=str(weights), num_classes=num_classes)
        if variant == "fp16":
            model.inference(compile=False, inplace=True, dtype="float16")  # tidak dapat dibalik; hanya untuk inference
        return TorchPredictor(variant, model)
    if variant == "int8":
        return OrtPredictor(variant, Path(fold_dir) / "variants/int8/model_int8.onnx")
    if variant == "prune_structured":
        import torch
        from rfdetr import RFDETRSegSmall

        from compress import shrink_like

        model = RFDETRSegSmall(pretrain_weights=None, num_classes=num_classes)
        sd = torch.load(Path(fold_dir) / "variants/prune_structured/weights.pth", map_location="cpu", weights_only=False)["model"]
        shrink_like(model.model.model, sd)  # dimensi FFN mengecil; bentuk harus cocok persis
        model.model.model.load_state_dict(sd, strict=True)
        model.model.model.eval()
        return TorchPredictor(variant, model)
    raise NotImplementedError(variant)


# ---------------------------------------------------------------- pembantu murni

def aggregate(per_fold: list[dict]) -> dict:
    """{kunci: (mean, std sampel ddof=1)} antar fold, mengabaikan nan; satu nilai valid -> std 0."""
    out = {}
    for key in per_fold[0]:
        v = np.array([f[key] for f in per_fold], float)
        v = v[~np.isnan(v)]
        out[key] = (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0) if v.size else (float("nan"), float("nan"))
    return out


def remap_instances(inst: Instances, mapping: dict) -> Instances:
    """Ganti class_id lewat mapping; instance dengan kelas di luar mapping dibuang."""
    keep = np.array([int(c) in mapping for c in inst.class_id], bool)
    return Instances(inst.masks[keep], np.array([mapping[int(c)] for c in inst.class_id[keep]], int), inst.score[keep])


def gt_instances(api, img_id: int, cat_to_global: dict, size: tuple[int, int]) -> Instances:
    """Anotasi GT satu gambar sebagai Instances; kelas = indeks global. api = pycocotools COCO."""
    anns = [a for a in api.loadAnns(api.getAnnIds(imgIds=[img_id])) if a["category_id"] in cat_to_global]
    if not anns:
        return Instances(np.zeros((0, *size), bool), np.zeros(0, int), np.zeros(0, float))
    masks = np.stack([api.annToMask(a) for a in anns]).astype(bool)
    return Instances(masks, np.array([cat_to_global[a["category_id"]] for a in anns], int), np.ones(len(anns)))


def to_detections(inst: Instances):
    import supervision as sv

    if len(inst.class_id) == 0:
        return sv.Detections.empty()
    return sv.Detections(xyxy=sv.mask_to_xyxy(inst.masks).astype(float), mask=inst.masks, class_id=inst.class_id, confidence=inst.score)


def plot_confusion(cm: np.ndarray, class_names: list[str], out_png: Path) -> None:
    """Tulis confusion matrix: out_png (jumlah), <nama>_norm.png (dinormalisasi per baris GT), <nama>.csv."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    labels = list(class_names) + ["background"]
    np.savetxt(out_png.with_suffix(".csv"), cm, fmt="%d", delimiter=",", header=",".join(labels), comments="")
    rows = cm.sum(1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros(cm.shape, float), where=rows > 0)
    for suffix, mat, fmt in (("", cm, "d"), ("_norm", norm, ".2f")):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(max(5, 0.7 * n), max(4, 0.6 * n)))
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(n), labels, rotation=45, ha="right")
        ax.set_yticks(range(n), labels)
        ax.set_xlabel("prediksi")
        ax.set_ylabel("ground truth")
        for r in range(n):
            for c in range(n):
                ax.text(c, r, format(mat[r, c], fmt), ha="center", va="center", fontsize=7, color="white" if mat[r, c] > mat.max() / 2 else "black")
        fig.savefig(out_png.with_name(out_png.stem + suffix + ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_panel(image, gt: Instances, preds: dict, out_png: Path) -> None:
    """Satu baris: Ground truth | tiap varian. Nilai None = varian GAGAL (panel kosong berlabel)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import supervision as sv

    cols = [("ground truth", gt)] + list(preds.items())
    fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 3.2))
    base = np.array(image.convert("RGB"))
    for ax, (name, inst) in zip(np.atleast_1d(axes), cols):
        ax.axis("off")
        if inst is None:
            ax.text(0.5, 0.5, "GAGAL", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name, fontsize=9)
            continue
        det = to_detections(inst)
        ax.imshow(sv.MaskAnnotator(opacity=0.5).annotate(base.copy(), det) if len(det) else base)
        ax.set_title(f"{name} ({len(inst.class_id)})", fontsize=9)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- biaya komputasi

class _GpuSampler:
    """Sampel memori terpakai (puncak, MB) dan utilisasi (rerata, %) GPU 0 lewat pynvml selama blok `with`."""

    def __enter__(self):
        self.mem, self.util, self._stop, self.ok = [], [], threading.Event(), False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv, self._h, self.ok = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0), True
        except Exception:  # noqa: BLE001 - tanpa GPU/pynvml: kolom dilaporkan None
            return self
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            self.mem.append(self._nv.nvmlDeviceGetMemoryInfo(self._h).used / 2**20)
            self.util.append(self._nv.nvmlDeviceGetUtilizationRates(self._h).gpu)
            time.sleep(0.02)

    def __exit__(self, *exc):
        if self.ok:
            self._stop.set()
            self._t.join()
        return False

    @property
    def peak_mb(self):
        return max(self.mem) if self.mem else None

    @property
    def util_mean(self):
        return float(np.mean(self.util)) if self.util else None


def measure_cost(predictor, images: list, warmup: int = 10, n: int = 100, threshold: float = 0.5) -> dict:
    """Latency (rerata, p50, p95; ms), FPS, memori GPU puncak (MB), utilisasi rerata (%); batch 1, predict() ujung-ke-ujung."""
    import torch
    from PIL import Image

    imgs = [Image.open(p).convert("RGB") for p in images]
    if not imgs:
        raise ValueError("tidak ada gambar untuk mengukur biaya komputasi")
    seq = [imgs[i % len(imgs)] for i in range(warmup + n)]
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for im in seq[:warmup]:
        predictor.predict(im, threshold)
    times = []
    with _GpuSampler() as gpu:
        for im in seq[warmup:]:
            sync()
            t0 = time.perf_counter()
            predictor.predict(im, threshold)
            sync()
            times.append((time.perf_counter() - t0) * 1000)
    return {
        "latency_ms_mean": float(np.mean(times)), "latency_ms_p50": float(np.percentile(times, 50)),
        "latency_ms_p95": float(np.percentile(times, 95)), "fps": 1000 / float(np.mean(times)),
        "gpu_mem_peak_mb": gpu.peak_mb, "gpu_util_mean_pct": gpu.util_mean,
    }


def count_gflops(module, resolution: int) -> float:
    """GFLOPs forward 1x3xRxR (torch FlopCounterMode; operator custom bisa tidak terhitung -> hanya untuk perbandingan relatif)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    module = module.float().eval()
    x = torch.randn(1, 3, resolution, resolution, device=next(module.parameters()).device)
    with FlopCounterMode(display=False) as fc:  # sengaja tanpa no_grad: FlopCounterMode di torch 2.x gagal pada model RF-DETR di bawah no_grad
        module(x)
    return fc.get_total_flops() / 1e9


# ---------------------------------------------------------------- orkestrasi

VARIANTS = ["fp32", "fp16", "int8", "prune_unstructured", "prune_structured"]
LOW_THR = 0.01  # threshold prediksi mentah untuk mAP; pencocokan/confusion memakai --threshold
METRIC_KEYS = ["map50_95", "map50", "precision_micro", "recall_micro", "f1_micro", "precision_macro", "recall_macro", "f1_macro", "accuracy_detection", "accuracy_classification"]


def load_context(out: Path) -> dict:
    """Dataset gabungan, fold, dan kelas global (urutan label rfdetr pada dataset gabungan)."""
    from pycocotools.coco import COCO

    from train import fold_classes

    coco = json.loads((out / "merged" / "_annotations.coco.json").read_text(encoding="utf-8"))
    api = COCO()
    api.dataset = coco
    with contextlib.redirect_stdout(io.StringIO()):
        api.createIndex()
    global_cats = fold_classes(coco)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    return {
        "coco_json": out / "merged" / "_annotations.coco.json", "images_dir": out / "merged" / "images", "api": api,
        "images": {im["id"]: im for im in coco["images"]}, "folds": json.loads((out / "folds.json").read_text(encoding="utf-8"))["folds"],
        "global_cats": global_cats, "cat_to_global": {c: i for i, c in enumerate(global_cats)}, "class_names": [names[c] for c in global_cats],
    }


def vis_ids(fold_ids: list[int], k: int, seed: int, n: int) -> list[int]:
    """Gambar visualisasi dipilih dengan seed tetap (sama untuk semua varian), bukan dipilih manual."""
    return sorted(random.Random(seed * 1000 + k).sample(fold_ids, min(n, len(fold_ids))))


def weights_path(variant: str, fold_dir: Path) -> Path:
    if variant == "int8":
        return fold_dir / "variants/int8/model_int8.onnx"
    if variant.startswith("prune"):
        return fold_dir / "variants" / variant / "weights.pth"
    return fold_dir / "weights.pth"


def artifact_problem(variant: str, fold_dir: Path) -> str | None:
    """Alasan varian tidak bisa dievaluasi (artefak gagal/belum ada), atau None."""
    if not (fold_dir / "DONE").exists():
        return f"{fold_dir}/DONE tidak ada (train.py belum selesai)"
    if variant in ("fp32", "fp16"):
        return None
    vd = fold_dir / "variants" / variant
    if (vd / "FAILED.txt").exists():
        return "compress.py gagal: " + (vd / "FAILED.txt").read_text(encoding="utf-8").splitlines()[0]
    return None if (vd / "DONE").exists() else f"{vd}/DONE tidak ada (jalankan compress.py)"


def variant_cost(variant: str, pred, fold_dir: Path, image_paths: list, out: Path, n: int) -> dict:
    """Biaya komputasi satu varian (fold 0): latency/FPS/GPU, GFLOPs, ukuran model, parameter."""
    import torch

    cost = measure_cost(pred, image_paths, n=n)
    wp = weights_path(variant, fold_dir)
    cost["file_mb"] = wp.stat().st_size / 2**20
    if variant == "int8":
        import onnx

        cost["params"] = int(sum(int(np.prod(i.dims)) for i in onnx.load(str(wp)).graph.initializer))
        cost["nonzero_params"] = None
        cost["provider"] = ",".join(pred.providers)
        cost["gflops"], cost["gflops_note"] = None, "sama dengan fp32 (jumlah operasi tidak berubah)"
    elif variant == "fp16":  # inference(inplace=True) mengosongkan modul PyTorch: angka arsitektur diambil dari fp32
        base = _fp32_cost(out)
        cost["params"], cost["nonzero_params"] = base.get("params"), base.get("nonzero_params")
        cost["file_mb"] = cost["params"] * 2 / 2**20 if cost["params"] else None  # bobot fp16 = 2 byte/parameter
        cost["provider"] = "cuda" if torch.cuda.is_available() else "cpu"
        cost["gflops"], cost["gflops_note"] = None, "sama dengan fp32 (jumlah operasi tidak berubah)"
    else:
        module = pred.model.model.model
        cost["params"] = int(sum(p.numel() for p in module.parameters()))
        cost["nonzero_params"] = int(sum((p != 0).sum().item() for p in module.parameters()))
        cost["provider"] = "cuda" if torch.cuda.is_available() else "cpu"
        if variant in ("fp32", "prune_structured"):  # arsitektur berbeda hanya pada prune_structured
            cost["gflops"], cost["gflops_note"] = count_gflops(module, pred.model.model_config.resolution), "diukur"
        else:
            cost["gflops"], cost["gflops_note"] = None, "sama dengan fp32 (jumlah operasi tidak berubah)"
    if variant == "prune_unstructured":
        cost["sparsity_pct"] = 100 * json.loads((fold_dir / "variants" / variant / "info.json").read_text(encoding="utf-8"))["sparsity"]
    if cost["gflops"] is None:  # salin dari fp32 bila tersedia
        cost["gflops"] = _fp32_cost(out).get("gflops")
    return cost


def _fp32_cost(out: Path) -> dict:
    f32 = out / "results" / "fp32" / "fold0.json"
    return json.loads(f32.read_text(encoding="utf-8")).get("cost", {}) if f32.exists() else {}


def evaluate_variant_fold(variant: str, k: int, ctx: dict, args) -> dict:
    """Evaluasi satu varian pada fold uji k; mengembalikan hasil (cm, metrik, mAP) dan, untuk fold 0, biaya komputasi."""
    import pycocotools.mask as mu
    from PIL import Image

    fold_dir = args.out / f"fold{k}"
    classes_k = json.loads((fold_dir / "classes.json").read_text(encoding="utf-8"))
    label_to_global = {lab: ctx["cat_to_global"][cat] for lab, cat in enumerate(classes_k)}
    C = len(ctx["global_cats"])
    pred = load_predictor(variant, fold_dir, len(classes_k))
    cm, coco_preds = np.zeros((C + 1, C + 1), int), []
    chosen = set(vis_ids(ctx["folds"][k], k, args.seed, args.n_visual))
    vis_dir = args.out / "results" / variant / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for img_id in ctx["folds"][k]:
        image = Image.open(ctx["images_dir"] / ctx["images"][img_id]["file_name"]).convert("RGB")
        size = (image.height, image.width)
        inst = remap_instances(to_instances(pred.predict(image, LOW_THR), size, len(classes_k)), label_to_global)
        for m, c, sc in zip(inst.masks, inst.class_id, inst.score):
            rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            coco_preds.append({"image_id": img_id, "category_id": ctx["global_cats"][int(c)], "segmentation": rle, "score": float(sc)})
        gt = gt_instances(ctx["api"], img_id, ctx["cat_to_global"], size)
        hi = Instances(inst.masks[inst.score >= args.threshold], inst.class_id[inst.score >= args.threshold], inst.score[inst.score >= args.threshold])
        cm += confusion_matrix(match_instances(gt, hi, score_thr=args.threshold), gt, hi, C)
        if img_id in chosen:
            np.savez_compressed(vis_dir / f"fold{k}_img{img_id}.npz", masks=hi.masks, class_id=hi.class_id, score=hi.score)
    summ = summarize(cm)
    mp = coco_map(ctx["coco_json"], coco_preds, ctx["folds"][k])
    result = {
        "status": "ok", "fold": k, "cm": cm.tolist(), "per_class": {str(c): v for c, v in summ["per_class"].items()},
        "ap_per_class": {str(c): v for c, v in mp["ap_per_class"].items()},
        "metrics": {**{key: summ[key] for key in METRIC_KEYS if key in summ}, "map50_95": mp["map50_95"], "map50": mp["map50"]},
    }
    if k == 0:
        paths = [ctx["images_dir"] / ctx["images"][i]["file_name"] for i in ctx["folds"][0]]
        try:  # kegagalan pengukuran biaya tidak boleh membuang metrik yang sudah dihitung
            result["cost"] = variant_cost(variant, pred, fold_dir, paths, args.out, args.cost_images)
        except Exception as e:  # noqa: BLE001
            result["cost_error"] = f"{type(e).__name__}: {e}"
            print(f"{variant}: biaya komputasi gagal diukur ({result['cost_error']})")
    return result


def write_summary(args, ctx: dict, variants: list[str]) -> Path:
    """summary.csv (mean +- std antar fold, biaya komputasi) dan confusion matrix gabungan per varian."""
    rows = []
    for v in variants:
        files = sorted((args.out / "results" / v).glob("fold*.json"))
        results = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        ok = [r for r in results if r["status"] == "ok"]
        row = {"variant": v}
        if not ok:
            row.update(status="GAGAL", reason=(results[0].get("reason", "belum dijalankan") if results else "belum dijalankan"))
            rows.append(row)
            continue
        row.update(status="ok" if len(ok) == args.folds else f"parsial {len(ok)}/{args.folds}", folds_ok=len(ok))
        for key, (mean, std) in aggregate([r["metrics"] for r in ok]).items():
            row[f"{key}_mean"], row[f"{key}_std"] = mean, std
        row.update({k: val for k, val in next((r["cost"] for r in ok if "cost" in r), {}).items()})
        err = next((r["cost_error"] for r in ok if "cost_error" in r), None)
        if err:
            row["cost_error"] = err
        pooled = np.sum([np.array(r["cm"]) for r in ok], axis=0)
        plot_confusion(pooled, ctx["class_names"], args.out / "results" / v / "confusion.png")
        rows.append(row)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    path = args.out / "results" / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def make_panels(args, ctx: dict, variants: list[str]) -> int:
    """Panel GT vs prediksi semua varian untuk gambar visualisasi tiap fold; varian GAGAL -> panel kosong."""
    from PIL import Image

    n = 0
    for k in range(args.folds):
        for img_id in vis_ids(ctx["folds"][k], k, args.seed, args.n_visual):
            image = Image.open(ctx["images_dir"] / ctx["images"][img_id]["file_name"]).convert("RGB")
            size = (image.height, image.width)
            preds = {}
            for v in variants:
                f = args.out / "results" / v / "vis" / f"fold{k}_img{img_id}.npz"
                if f.exists():
                    d = np.load(f)
                    preds[v] = Instances(d["masks"], d["class_id"], d["score"])
                elif (args.out / "results" / v / f"fold{k}.json").exists():
                    preds[v] = None  # varian dijalankan tetapi gagal
            if preds:
                save_panel(image, gt_instances(ctx["api"], img_id, ctx["cat_to_global"], size), preds, args.out / "results" / "panels" / f"fold{k}_img{img_id}.png")
                n += 1
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5, help="threshold skor untuk confusion matrix/precision/recall")
    ap.add_argument("--n-visual", type=int, default=8, help="gambar visualisasi per fold")
    ap.add_argument("--cost-images", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 fold")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args(argv)
    if args.smoke:
        args.folds = 1
    import gc

    import torch

    if not torch.cuda.is_available():
        print("PERINGATAN: GPU tidak terdeteksi; latency/memori GPU tidak bermakna (kolom biaya diisi apa adanya dari CPU).")
    ctx = load_context(args.out)
    for v in args.variants:
        for k in range(args.folds):
            rp = args.out / "results" / v / f"fold{k}.json"
            if rp.exists() and json.loads(rp.read_text(encoding="utf-8"))["status"] == "ok":
                print(f"{v} fold {k}: sudah ada, dilewati")
                continue
            try:
                problem = artifact_problem(v, args.out / f"fold{k}")
                if problem:
                    raise RuntimeError(problem)
                result = evaluate_variant_fold(v, k, ctx, args)
                print(f"{v} fold {k}: ok " + " ".join(f"{m}={result['metrics'][m]:.3f}" for m in ("map50_95", "f1_micro")))
            except Exception as e:  # noqa: BLE001 - varian gagal ditandai, varian lain tetap jalan
                result = {"status": "GAGAL", "fold": k, "reason": f"{type(e).__name__}: {e}"}
                print(f"{v} fold {k}: GAGAL ({result['reason']})")
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(result), encoding="utf-8")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("ringkasan:", write_summary(args, ctx, args.variants), "| panel:", make_panels(args, ctx, args.variants))


if __name__ == "__main__":
    main()
