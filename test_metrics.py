"""Pemeriksaan murni (tanpa GPU): `python test_metrics.py`. Fungsi test_* juga bisa dijalankan pytest."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from train import load_or_create_folds, make_folds, merge_coco, write_fold_dir


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
