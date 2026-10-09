"""Pemeriksaan murni (tanpa GPU): `python test_metrics.py`. Fungsi test_* juga bisa dijalankan pytest."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate import Instances, aggregate, panel_state, result_is_current, write_summary, coco_map, count_gflops, gt_instances, plot_confusion, remap_instances, save_panel, confusion_matrix, match_instances, load_predictor, summarize, to_instances
from train import done_matches, load_env, resolve_out, write_done, load_or_create_folds, make_folds, merge_coco, write_fold_dir


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


def test_prune_ffn_pair_shapes_and_output():
    import torch
    import torch.nn as nn

    from compress import prune_ffn_pair

    torch.manual_seed(0)
    l1, l2 = nn.Linear(8, 16), nn.Linear(16, 8)
    with torch.no_grad():
        dead = [1, 5, 9, 14]  # neuron "mati": bobot masuk dan keluar nol -> paling tidak penting
        l1.weight[dead] = 0
        l1.bias[dead] = 0
        l2.weight[:, dead] = 0
    n1, n2 = prune_ffn_pair(l1, l2, ratio=0.25)
    assert n1.weight.shape == (12, 8) and n1.bias.shape == (12,) and n2.weight.shape == (8, 12)
    x = torch.randn(5, 8)
    assert torch.allclose(n2(torch.relu(n1(x))), l2(torch.relu(l1(x))), atol=1e-6)
    assert torch.equal(n2.bias, l2.bias)


def test_prune_ffn_finds_both_naming_schemes():
    import torch.nn as nn

    from compress import prune_ffn

    class Layer(nn.Module):
        def __init__(self, a, b, d=8, h=16):
            super().__init__()
            setattr(self, a, nn.Linear(d, h))
            setattr(self, b, nn.Linear(h, d))

    net = nn.ModuleDict({"dec": Layer("linear1", "linear2"), "mlp": Layer("fc1", "fc2"), "other": nn.Linear(8, 8)})
    kept = prune_ffn(net, ratio=0.5)
    assert kept == 0.5
    assert net["dec"].linear1.out_features == 8 and net["dec"].linear2.in_features == 8
    assert net["mlp"].fc1.out_features == 8 and net["mlp"].fc2.in_features == 8
    assert net["other"].out_features == 8


def test_pruned_real_model_reloads_via_shrink_like():
    try:
        import torch
        from rfdetr import RFDETRSegSmall
    except ImportError:
        print("SKIP test_pruned_real_model_reloads_via_shrink_like (rfdetr tidak terpasang)")
        return
    from compress import prune_ffn, shrink_like

    mod = RFDETRSegSmall(pretrain_weights=None, device="cpu", num_classes=3).model.model
    full_params = sum(p.numel() for p in mod.parameters())
    kept = prune_ffn(mod, ratio=0.25)
    assert abs(kept - 0.75) < 0.01
    sd = {k: v.clone() for k, v in mod.state_dict().items()}
    assert sum(p.numel() for p in mod.parameters()) < full_params
    fresh = RFDETRSegSmall(pretrain_weights=None, device="cpu", num_classes=3).model.model
    shrink_like(fresh, sd)
    fresh.load_state_dict(sd, strict=True)  # bentuk harus cocok persis


def test_structured_variant_loads_and_predicts_on_cpu():
    try:
        import torch
        from rfdetr import RFDETRSegSmall
    except ImportError:
        print("SKIP test_structured_variant_loads_and_predicts_on_cpu (rfdetr tidak terpasang)")
        return
    from compress import prune_ffn

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        mod = RFDETRSegSmall(pretrain_weights=None, device="cpu", num_classes=3).model.model
        prune_ffn(mod, 0.25)
        dest = t / "variants" / "prune_structured"
        dest.mkdir(parents=True)
        torch.save({"model": mod.state_dict()}, dest / "weights.pth")
        pred = load_predictor("prune_structured", t, num_classes=3)
        assert pred.name == "prune_structured"
        det = pred.predict(Image.new("RGB", (96, 64), (90, 90, 90)), threshold=0.0)
        inst = to_instances(det, (64, 96), num_classes=3)
        assert inst.masks.shape[1:] == (64, 96)  # mask berukuran gambar asli, bukan 640x640
        assert inst.class_id.size == 0 or inst.class_id.max() < 3


def test_to_instances_drops_background_slot():
    import supervision as sv

    m = rect_masks(A, B, C_, size=10)
    det = sv.Detections(xyxy=np.zeros((3, 4)), mask=m, class_id=np.array([0, 3, 2]), confidence=np.array([0.9, 0.8, 0.7]))
    got = to_instances(det, (10, 10), num_classes=3)  # class_id == num_classes adalah slot no-object rfdetr
    assert got.class_id.tolist() == [0, 2] and got.score.tolist() == [0.9, 0.7]
    assert np.array_equal(got.masks, m[[0, 2]])
    only_bg = sv.Detections(xyxy=np.zeros((1, 4)), mask=m[:1], class_id=np.array([3]), confidence=np.array([0.5]))
    assert to_instances(only_bg, (10, 10), num_classes=3).masks.shape == (0, 10, 10)


def test_decode_masks_resizes_and_thresholds_logits():
    from evaluate import decode_masks

    logits = np.full((1, 4, 4), -5.0, np.float32)
    logits[0, :2, :2] = 5.0  # kuadran kiri-atas positif
    m = decode_masks(logits, (8, 8))
    assert m.shape == (1, 8, 8) and m.dtype == bool
    assert m[0, :3, :3].all() and not m[0, 5:, 5:].any()
    assert decode_masks(np.zeros((0, 4, 4), np.float32), (6, 7)).shape == (0, 6, 7)


def test_calibration_reader_yields_expected_shape():
    try:
        import rfdetr  # noqa: F401  (preprocess_to_nchw berasal dari rfdetr)
    except ImportError:
        print("SKIP test_calibration_reader_yields_expected_shape (rfdetr tidak terpasang)")
        return
    from compress import ImageCalibrationReader

    with tempfile.TemporaryDirectory() as t:
        paths = []
        for i in range(3):
            p = Path(t) / f"c{i}.jpg"
            Image.new("RGB", (20 + i, 12), (i * 50, 20, 20)).save(p)
            paths.append(p)
        reader = ImageCalibrationReader(paths, "input", 16, 16)
        got = []
        while (batch := reader.get_next()) is not None:
            got.append(batch)
        assert len(got) == 3
        assert all(list(b) == ["input"] and b["input"].shape == (1, 3, 16, 16) and b["input"].dtype == np.float32 for b in got)
        reader.rewind()
        assert reader.get_next() is not None  # bisa diulang (kuantisasi membaca data lebih dari sekali)


def test_remap_instances_drops_unknown_labels():
    i = inst([A, B, C_], [0, 1, 2], scores=[0.9, 0.8, 0.7])
    got = remap_instances(i, {0: 5, 2: 7})  # label 1 tidak dikenal -> dibuang
    assert got.class_id.tolist() == [5, 7] and got.score.tolist() == [0.9, 0.7]
    assert np.array_equal(got.masks, i.masks[[0, 2]])
    assert remap_instances(inst([], []), {0: 1}).masks.shape == (0, 10, 10)


def test_aggregate_mean_std_ignores_nan():
    agg = aggregate([{"f1": 0.2, "p": float("nan")}, {"f1": 0.4, "p": float("nan")}, {"f1": 0.6, "p": 1.0}])
    assert abs(agg["f1"][0] - 0.4) < 1e-9 and abs(agg["f1"][1] - 0.2) < 1e-9  # std sampel (ddof=1)
    assert agg["p"] == (1.0, 0.0)  # satu nilai valid: std 0
    assert np.isnan(aggregate([{"x": float("nan")}])["x"][0])


def test_gt_instances_from_polygons_and_empty():
    from pycocotools.coco import COCO

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}, {"id": 2, "file_name": "b.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "iscrowd": 0, "area": 4, "bbox": [0, 0, 2, 2], "segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]]}],
        "categories": [{"id": 7, "name": "x"}],
    }
    api = COCO()
    api.dataset = coco
    api.createIndex()
    g = gt_instances(api, 1, {7: 3}, (10, 10))
    assert g.class_id.tolist() == [3] and g.masks.shape == (1, 10, 10) and g.masks[0, :2, :2].all() and g.masks.sum() < 10
    empty = gt_instances(api, 2, {7: 3}, (10, 10))
    assert empty.masks.shape == (0, 10, 10) and len(empty.class_id) == 0


def test_plot_confusion_writes_png_and_csv():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "cm.png"
        plot_confusion(np.array([[3, 1, 0], [0, 2, 1], [1, 0, 0]]), ["a", "b"], out)
        assert out.stat().st_size > 0 and (Path(t) / "cm_norm.png").stat().st_size > 0 and (Path(t) / "cm.csv").exists()
        plot_confusion(np.zeros((3, 3), int), ["a", "b"], Path(t) / "z.png")  # matriks nol tidak boleh crash


def test_save_panel_with_empty_and_failed_variants():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "panel.png"
        img = Image.new("RGB", (10, 10), (120, 120, 120))
        save_panel(img, inst([A], [0]), {"fp32": inst([A, B], [0, 1]), "int8": inst([], []), "prune_structured": None}, out)
        assert out.stat().st_size > 0


def test_count_gflops_matches_hand_computation():
    import torch.nn as nn

    conv = nn.Conv2d(3, 4, 3, bias=False)  # keluaran 4x6x6 = 144, tiap keluaran 27 MAC -> 3888 MAC = 7776 FLOP
    assert abs(count_gflops(conv, 8) * 1e9 - 7776) < 1


def test_count_gflops_on_real_model_and_pruning_reduces_it():
    try:
        from rfdetr import RFDETRSegSmall
    except ImportError:
        print("SKIP test_count_gflops_on_real_model_and_pruning_reduces_it (rfdetr tidak terpasang)")
        return
    from compress import prune_ffn

    m = RFDETRSegSmall(pretrain_weights=None, device="cpu", num_classes=3)
    res = m.model_config.resolution
    full = count_gflops(m.model.model, res)
    assert 10 < full < 500, full  # RFDETRSegSmall ~63 GFLOPs pada 384
    prune_ffn(m.model.model, 0.3)
    assert count_gflops(m.model.model, res) < full  # pruning terstruktur benar-benar menurunkan GFLOPs


def test_resolve_out_keeps_smoke_apart_from_full_run():
    assert resolve_out(None, smoke=False) == Path("outputs")
    assert resolve_out(None, smoke=True) == Path("outputs_smoke")  # smoke tidak boleh menimpa/dipakai ulang oleh run penuh
    assert resolve_out(Path("x"), smoke=True) == Path("x")  # --out eksplisit selalu menang


def test_done_stamp_matches_only_identical_settings():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "DONE"
        assert not done_matches(p, {"epochs": 50})  # belum ada
        write_done(p, {"epochs": 50, "seed": 0})
        assert done_matches(p, {"epochs": 50, "seed": 0})
        assert not done_matches(p, {"epochs": 1, "seed": 0})  # pengaturan beda (mis. smoke 1 epoch)
        p.write_text("ok")  # penanda lama tanpa stempel
        assert not done_matches(p, {"epochs": 50, "seed": 0})


def test_result_is_current_requires_same_stamp_and_clean_cost():
    stamp = {"threshold": 0.5, "seed": 0}
    ok = {"status": "ok", "stamp": stamp, "cost": {"fps": 1.0}}
    assert result_is_current(ok, stamp, need_cost=True)
    assert not result_is_current(None, stamp, need_cost=False)
    assert not result_is_current({**ok, "stamp": {"threshold": 0.3, "seed": 0}}, stamp, need_cost=False)  # pengaturan berubah
    assert not result_is_current({"status": "GAGAL", "stamp": stamp}, stamp, need_cost=False)
    assert not result_is_current({"status": "ok", "stamp": stamp}, stamp, need_cost=True)  # fold 0 tanpa biaya diukur ulang
    assert not result_is_current({**ok, "cost_error": "boom"}, stamp, need_cost=True)
    assert result_is_current({"status": "ok", "stamp": stamp}, stamp, need_cost=False)  # fold > 0 tidak butuh biaya


def test_panel_state():
    assert panel_state("ok", True) == "show"
    assert panel_state("ok", False) == "skip"  # gambar visual berubah (seed/n_visual): bukan kegagalan varian
    assert panel_state("GAGAL", False) == "failed"
    assert panel_state(None, False) == "skip"  # varian belum dijalankan


def test_write_summary_pooled_per_class_and_fp32_cost_copy():
    import csv
    from types import SimpleNamespace

    def res(cm, cost=None):
        m = {k: 0.5 for k in ("map50_95", "map50", "precision_micro", "recall_micro", "f1_micro", "precision_macro", "recall_macro", "f1_macro", "accuracy_detection", "accuracy_classification")}
        r = {"status": "ok", "fold": 0, "cm": cm, "metrics": m, "ap_per_class": {"1": 0.5, "2": 0.25}}
        if cost is not None:
            r["cost"] = cost
        return r

    cm = [[2, 0, 1], [0, 1, 1], [1, 0, 0]]
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        for v, cost in (("fp32", {"gflops": 62.0, "params": 100, "nonzero_params": 90, "fps": 2.0}), ("fp16", {"gflops": None, "params": None, "nonzero_params": None, "fps": 4.0})):
            (out / "results" / v).mkdir(parents=True)
            (out / "results" / v / "fold0.json").write_text(json.dumps(res(cm, cost)))
        ctx = {"class_names": ["a", "b"], "global_cats": [1, 2]}
        path = write_summary(SimpleNamespace(out=out, folds=1), ctx, ["fp32", "fp16"])
        rows = {r["variant"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
        assert float(rows["fp16"]["gflops"]) == 62.0 and float(rows["fp16"]["params"]) == 100.0  # disalin dari fp32
        assert "latency_scope" in rows["fp16"]  # catatan cakupan latency (end-to-end predict())
        assert abs(float(rows["fp32"]["pooled_precision_micro"]) - 3 / 4) < 1e-9  # TP=3, FP=1 dari cm gabungan
        pc = list(csv.DictReader(open(out / "results" / "fp32" / "per_class.csv", encoding="utf-8")))
        assert [r["class"] for r in pc] == ["a", "b"] and abs(float(pc[0]["ap_mean"]) - 0.5) < 1e-9


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
