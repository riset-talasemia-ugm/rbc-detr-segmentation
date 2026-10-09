"""Pemeriksaan murni (tanpa GPU): `python test_metrics.py`. Fungsi test_* juga bisa dijalankan pytest."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate import Instances, coco_map, confusion_matrix, match_instances, resolve_mapping, summarize, to_instances
from train import load_env, load_or_create_folds, make_folds, merge_coco, write_fold_dir


def synthetic_coco(n_images=10, class1_images=(0, 2, 4, 6, 8), with_empty=()):
    """n_images gambar; semuanya punya anotasi kelas 0 kecuali `with_empty`; kelas 1 hanya di `class1_images`."""
    cats = [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}]
    images = [{"id": i, "file_name": f"img{i}.jpg", "width": 8, "height": 8} for i in range(n_images)]
    anns = []
    for i in range(n_images):
        if i in with_empty:
            continue
        anns.append({"id": len(anns), "image_id": i, "category_id": 0, "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0, "segmentation": []})
        if i in class1_images:
            anns.append({"id": len(anns), "image_id": i, "category_id": 1, "bbox": [3, 3, 2, 2], "area": 4, "iscrowd": 0, "segmentation": []})
    return {"images": images, "annotations": anns, "categories": cats}


def _write_split(d: Path, names: list[str], cat_id: int) -> None:
    d.mkdir(parents=True)
    images, anns = [], []
    for i, n in enumerate(names):
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(d / n)
        images.append({"id": i, "file_name": n, "width": 8, "height": 8})
        anns.append({"id": i, "image_id": i, "category_id": cat_id, "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0, "segmentation": []})
    cats = [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}]
    (d / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))


def test_make_folds_disjoint_and_complete():
    coco = synthetic_coco()
    folds = make_folds(coco, k=5, seed=0)
    flat = [i for f in folds for i in f]
    assert sorted(flat) == list(range(10)) and len(set(flat)) == 10
    ann_cls = {a["image_id"]: set() for a in coco["annotations"]}
    for a in coco["annotations"]:
        ann_cls[a["image_id"]].add(a["category_id"])
    for f in folds:
        assert sum(1 for i in f if 1 in ann_cls[i]) == 1


def test_make_folds_deterministic():
    coco = synthetic_coco()
    assert make_folds(coco, 5, seed=3) == make_folds(coco, 5, seed=3)


def test_make_folds_image_without_annotations():
    coco = synthetic_coco(with_empty=(9,))
    flat = [i for f in make_folds(coco, 5, seed=0) for i in f]
    assert sorted(flat) == list(range(10))


def test_merge_coco_relinks_annotations():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        _write_split(t / "train", ["img.jpg", "x.jpg"], cat_id=0)  # id gambar/anotasi sama di kedua split
        _write_split(t / "valid", ["img.jpg"], cat_id=1)  # nama file bentrok
        merged = merge_coco([t / "train", t / "valid"], t / "out")
        assert len(merged["images"]) == 3 and len({im["id"] for im in merged["images"]}) == 3
        assert len({a["id"] for a in merged["annotations"]}) == 3
        by_id = {im["id"]: im for im in merged["images"]}
        for a in merged["annotations"]:
            assert a["image_id"] in by_id
        for im in merged["images"]:
            assert (t / "out" / "images" / im["file_name"]).exists()
        # anotasi kelas 1 berasal dari split valid: gambarnya harus berwarna (0,0,0) = gambar valid pertama
        a1 = next(a for a in merged["annotations"] if a["category_id"] == 1)
        px = Image.open(t / "out" / "images" / by_id[a1["image_id"]]["file_name"]).getpixel((0, 0))
        assert px == (0, 0, 0)
        assert (t / "out" / "_annotations.coco.json").exists()


def test_load_or_create_folds_rejects_mismatch():
    coco = synthetic_coco()
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "folds.json"
        created = load_or_create_folds(coco, k=5, seed=0, path=p)
        assert load_or_create_folds(coco, k=5, seed=0, path=p) == created
        for kw in ({"k": 3, "seed": 0}, {"k": 5, "seed": 1}):
            try:
                load_or_create_folds(coco, path=p, **kw)
            except SystemExit:
                continue
            raise AssertionError(f"SystemExit tidak muncul untuk {kw}")


def test_write_fold_dir_splits_by_fold():
    coco = synthetic_coco()
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "images").mkdir()
        for im in coco["images"]:
            Image.new("RGB", (8, 8)).save(t / "images" / im["file_name"])
        folds = make_folds(coco, 5, seed=0)
        out = write_fold_dir(coco, folds, val_fold=2, images_dir=t / "images", out_dir=t / "fold")
        valid = json.loads((out / "valid" / "_annotations.coco.json").read_text())
        train = json.loads((out / "train" / "_annotations.coco.json").read_text())
        assert sorted(im["id"] for im in valid["images"]) == sorted(folds[2])
        assert len(train["images"]) == 10 - len(folds[2])
        assert len(list((out / "valid").glob("*.jpg"))) == len(folds[2])


def rect_masks(*boxes, size=10):
    """boxes = (r0, r1, c0, c1) -> mask bool (N, size, size)."""
    m = np.zeros((len(boxes), size, size), bool)
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        m[i, r0:r1, c0:c1] = True
    return m


def inst(boxes, classes, scores=None, size=10):
    masks = rect_masks(*boxes, size=size) if boxes else np.zeros((0, size, size), bool)
    sc = np.ones(len(classes)) if scores is None else np.array(scores, float)
    return Instances(masks, np.array(classes, int), sc)


A, B, C_, D = (0, 2, 0, 2), (3, 5, 3, 5), (6, 8, 6, 8), (8, 10, 0, 2)


def test_confusion_and_metrics_by_hand():
    gt = inst([A, B, C_], [0, 1, 0])
    pred = inst([A, B, D], [0, 0, 1])  # A benar, B salah kelas, D tanpa GT, C tanpa prediksi
    cm = confusion_matrix(match_instances(gt, pred), gt, pred, num_classes=2)
    assert cm.tolist() == [[1, 0, 1], [1, 0, 0], [0, 1, 0]], cm
    s = summarize(cm)
    for k in ("precision_micro", "recall_micro", "f1_micro"):
        assert abs(s[k] - 1 / 3) < 1e-9, (k, s[k])
    assert abs(s["accuracy_detection"] - 0.2) < 1e-9
    assert abs(s["accuracy_classification"] - 0.5) < 1e-9


def test_empty_gt_and_empty_pred():
    gt0, pred0 = inst([], []), inst([], [])
    assert confusion_matrix(match_instances(gt0, pred0), gt0, pred0, 2).sum() == 0
    pred2 = inst([A, B], [0, 1])
    cm = confusion_matrix(match_instances(gt0, pred2), gt0, pred2, 2)
    assert cm.tolist() == [[0, 0, 0], [0, 0, 0], [1, 1, 0]], cm  # semua FP
    gt2 = inst([A, B], [0, 1])
    cm = confusion_matrix(match_instances(gt2, pred0), gt2, pred0, 2)
    assert cm.tolist() == [[0, 0, 1], [0, 0, 1], [0, 0, 0]], cm  # semua FN
    summarize(np.zeros((3, 3), int))  # tidak boleh exception


def test_class_without_instances_is_nan():
    gt = inst([A, B], [0, 1])
    pred = inst([A, B], [0, 1])
    s = summarize(confusion_matrix(match_instances(gt, pred), gt, pred, num_classes=3))
    assert np.isnan(s["per_class"][2]["precision"]) and np.isnan(s["per_class"][2]["recall"])
    assert s["precision_macro"] == 1.0 and s["recall_macro"] == 1.0


def test_score_threshold_drops_low_scores():
    gt = inst([A], [0])
    pred = inst([A], [0], scores=[0.4])
    cm = confusion_matrix(match_instances(gt, pred, score_thr=0.5), gt, pred, 1)
    assert cm.tolist() == [[0, 1], [0, 0]], cm  # GT jadi FN, tidak ada FP


def _rle(mask):
    import pycocotools.mask as mu

    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return r


def test_coco_map_perfect_and_empty():
    mask = rect_masks(A)[0]
    gt = {
        "images": [{"id": 1, "file_name": "x.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": _rle(mask), "area": int(mask.sum()), "bbox": [0, 0, 2, 2], "iscrowd": 0}],
        "categories": [{"id": 0, "name": "super"}, {"id": 1, "name": "a"}],
    }
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "gt.json"
        p.write_text(json.dumps(gt))
        perfect = coco_map(p, [{"image_id": 1, "category_id": 1, "segmentation": _rle(mask), "score": 0.9}], [1])
        assert abs(perfect["map50_95"] - 1.0) < 1e-6 and abs(perfect["map50"] - 1.0) < 1e-6
        assert abs(perfect["ap_per_class"][1] - 1.0) < 1e-6
        empty = coco_map(p, [], [1])
        assert empty["map50_95"] == 0.0 and empty["map50"] == 0.0


def test_mapping_skips_unannotated_supercategory():
    assert resolve_mapping([0, 1, 2], [1, 2], {0, 1}) == "index_real"
    # skor mAP per kandidat menentukan bila diberikan
    assert resolve_mapping([0, 1, 2], [1, 2], {0, 1}, scores={"direct": 0.5, "index_all": 0.1, "index_real": 0.2}) == "direct"
    # kelas model di luar jangkauan suatu mapping -> mapping itu tidak valid
    assert resolve_mapping([1, 2, 3], [1, 2, 3], {0, 1, 2}) == "index_real"


def test_load_env_requires_all_variables():
    full = {"ROBOFLOW_API_KEY": "k", "ROBOFLOW_WORKSPACE": "w", "ROBOFLOW_PROJECT": "p", "ROBOFLOW_VERSION": "8"}
    assert load_env(full) == {"api_key": "k", "workspace": "w", "project": "p", "version": 8}
    for missing in full:
        env = {**full, missing: ""}
        try:
            load_env(env)
        except SystemExit as e:
            assert missing in str(e)
            continue
        raise AssertionError(f"SystemExit tidak muncul saat {missing} kosong")
    try:
        load_env({**full, "ROBOFLOW_VERSION": "delapan"})
    except SystemExit as e:
        assert "ROBOFLOW_VERSION" in str(e)
    else:
        raise AssertionError("versi non-angka harus ditolak")


def test_to_instances_without_and_with_mask():
    import supervision as sv

    empty = to_instances(sv.Detections.empty(), (10, 12))
    assert empty.masks.shape == (0, 10, 12) and empty.masks.dtype == bool
    assert len(empty.class_id) == 0 and len(empty.score) == 0
    m = rect_masks(A, B, size=10)
    det = sv.Detections(xyxy=np.array([[0, 0, 2, 2], [3, 3, 5, 5]], float), mask=m, class_id=np.array([1, 0]), confidence=np.array([0.9, 0.6]))
    got = to_instances(det, (10, 10))
    assert np.array_equal(got.masks, m) and got.class_id.tolist() == [1, 0] and got.score.tolist() == [0.9, 0.6]
    no_mask = sv.Detections(xyxy=np.array([[0, 0, 2, 2]], float), class_id=np.array([1]), confidence=np.array([0.9]))
    assert to_instances(no_mask, (10, 10)).masks.shape == (0, 10, 10)  # tanpa mask = tidak ada instance bermask


def test_prune_state_dict_sparsity_and_exclusions():
    import torch
    import torch.nn as nn

    from compress import prune_state_dict

    torch.manual_seed(0)
    m = nn.ModuleDict({"fc": nn.Linear(32, 64), "refpoint_embed": nn.Embedding(10, 8), "fc2": nn.Linear(64, 10)})
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    pruned, sparsity = prune_state_dict(sd, 0.5)
    w = torch.cat([pruned["fc.weight"].flatten(), pruned["fc2.weight"].flatten()])
    assert abs(sparsity - 0.5) < 0.02 and abs((w == 0).float().mean().item() - sparsity) < 1e-9
    assert torch.equal(pruned["refpoint_embed.weight"], sd["refpoint_embed.weight"])  # embedding tidak dipangkas
    assert torch.equal(pruned["fc.bias"], sd["fc.bias"])  # bias tidak dipangkas
    assert (sd["fc.weight"] == 0).sum() == 0  # input tidak dimutasi


def test_prune_unstructured_keeps_checkpoint_format():
    import torch
    import torch.nn as nn

    from compress import prune_unstructured

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        torch.manual_seed(0)
        sd = nn.Linear(16, 16).state_dict()
        torch.save({"model": sd, "args": {"epochs": 1}, "model_config": {"x": 1}}, t / "in.pth")
        sparsity = prune_unstructured(t / "in.pth", t / "sub" / "out.pth", amount=0.5)
        out = torch.load(t / "sub" / "out.pth", weights_only=False)
        assert out["args"] == {"epochs": 1} and out["model_config"] == {"x": 1}
        assert abs((out["model"]["weight"] == 0).float().mean().item() - sparsity) < 1e-9 and sparsity > 0.4


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} lolos")
    sys.exit(1 if failed else 0)
