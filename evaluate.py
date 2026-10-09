"""Evaluasi semua varian: metrik, confusion matrix, biaya komputasi, visualisasi, summary.csv.

Impor berat (torch, rfdetr, onnxruntime) dilakukan di dalam fungsi yang memakainya agar fungsi murni bisa diuji tanpa GPU.
"""
from dataclasses import dataclass

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
