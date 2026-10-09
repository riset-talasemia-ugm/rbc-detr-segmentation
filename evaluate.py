"""Evaluasi semua varian: metrik, confusion matrix, biaya komputasi, visualisasi, summary.csv.

Impor berat (torch, rfdetr, onnxruntime) dilakukan di dalam fungsi yang memakainya agar fungsi murni bisa diuji tanpa GPU.
"""
import contextlib
import io
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


# class_id model -> category_id GT; mana yang benar bergantung pada cara rfdetr memberi nomor kelas, jadi dicari otomatis.
MAPPINGS = {
    "direct": lambda c, cat_ids, real_ids: c if c in cat_ids else None,
    "index_all": lambda c, cat_ids, real_ids: cat_ids[c] if 0 <= c < len(cat_ids) else None,
    "index_real": lambda c, cat_ids, real_ids: real_ids[c] if 0 <= c < len(real_ids) else None,
}


def resolve_mapping(cat_ids: list[int], real_ids: list[int], raw_class_ids: set[int], scores: dict[str, float] | None = None) -> str:
    """Pilih mapping yang memetakan semua class_id terprediksi. Bila ada beberapa kandidat: yang mAP-nya tertinggi
    (scores, nama -> mAP), atau index_real bila scores tidak diberikan."""
    valid = [n for n, f in MAPPINGS.items() if all(f(c, cat_ids, real_ids) is not None for c in raw_class_ids)]
    if not valid:
        raise ValueError(f"Tidak ada mapping yang cocok untuk class_id {sorted(raw_class_ids)} (kategori {cat_ids})")
    if scores:
        return max(valid, key=lambda n: scores.get(n, float("-inf")))
    return "index_real" if "index_real" in valid else valid[0]


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
