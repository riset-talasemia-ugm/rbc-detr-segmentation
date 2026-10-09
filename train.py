"""Unduh dataset Roboflow, gabungkan semua split, bagi 5 fold terstratifikasi, latih satu model per fold."""
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ANN_NAME = "_annotations.coco.json"


def merge_coco(src_dirs: list[Path], out_dir: Path) -> dict:
    """Gabungkan split COCO ke satu dataset: gambar ke out_dir/images, id gambar dan anotasi di-re-id."""
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    merged = {"images": [], "annotations": [], "categories": None}
    used = set()
    for di, src in enumerate(src_dirs):
        src = Path(src)
        coco = json.loads((src / ANN_NAME).read_text(encoding="utf-8"))
        if merged["categories"] is None:
            merged["categories"] = coco["categories"]
        elif coco["categories"] != merged["categories"]:
            raise ValueError(f"Kategori {src} berbeda dari split sebelumnya")
        id_map = {}
        for im in coco["images"]:
            name = im["file_name"]
            if name in used:  # nama bentrok antar split: beri awalan indeks split
                name = f"{di}_{name}"
            used.add(name)
            shutil.copy2(src / im["file_name"], images_out / name)
            new_id = len(merged["images"])
            id_map[im["id"]] = new_id
            merged["images"].append({**im, "id": new_id, "file_name": name})
        for a in coco["annotations"]:
            merged["annotations"].append({**a, "id": len(merged["annotations"]), "image_id": id_map[a["image_id"]]})
    (out_dir / ANN_NAME).write_text(json.dumps(merged), encoding="utf-8")
    return merged


def make_folds(coco: dict, k: int, seed: int) -> list[list[int]]:
    """Id gambar per fold. Greedy: gambar dengan kelas paling langka ditempatkan lebih dulu
    ke fold yang saat itu paling sedikit memuat kelas langka itu."""
    classes = defaultdict(set)  # image id -> set kelas
    for im in coco["images"]:
        classes[im["id"]]
    for a in coco["annotations"]:
        classes[a["image_id"]].add(a["category_id"])
    freq = defaultdict(int)  # jumlah gambar per kelas
    for cs in classes.values():
        for c in cs:
            freq[c] += 1
    ids = sorted(classes)
    random.Random(seed).shuffle(ids)
    rarest = {i: min(classes[i], key=lambda c: (freq[c], c), default=None) for i in ids}
    ids.sort(key=lambda i: freq[rarest[i]] if rarest[i] is not None else float("inf"))  # stabil: urutan acak tetap untuk seri
    folds = [[] for _ in range(k)]
    count = [defaultdict(int) for _ in range(k)]  # fold -> kelas -> jumlah gambar
    for i in ids:
        r = rarest[i]
        f = min(range(k), key=lambda f: (count[f][r] if r is not None else 0, len(folds[f]), f))
        folds[f].append(i)
        for c in classes[i]:
            count[f][c] += 1
    return [sorted(f) for f in folds]


def write_fold_dir(coco: dict, folds: list[list[int]], val_fold: int, images_dir: Path, out_dir: Path) -> Path:
    """Tulis out_dir/train (semua fold selain val_fold) dan out_dir/valid (val_fold), lengkap dengan gambar."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    by_id = {im["id"]: im for im in coco["images"]}
    for split, ids in (("valid", folds[val_fold]), ("train", [i for f, fi in enumerate(folds) if f != val_fold for i in fi])):
        d = out_dir / split
        d.mkdir(parents=True)
        keep = set(ids)
        for i in ids:
            shutil.copy2(Path(images_dir) / by_id[i]["file_name"], d / by_id[i]["file_name"])
        sub = {
            "images": [by_id[i] for i in sorted(keep)],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
            "categories": coco["categories"],
        }
        (d / ANN_NAME).write_text(json.dumps(sub), encoding="utf-8")
    return out_dir


def load_or_create_folds(coco: dict, k: int, seed: int, path: Path) -> list[list[int]]:
    """Baca folds.json bila ada; berhenti jika k, seed, atau jumlah gambar berbeda (jangan campur hasil lama dan baru)."""
    path = Path(path)
    n = len(coco["images"])
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved["k"], saved["seed"], saved["n_images"]) != (k, seed, n):
            raise SystemExit(
                f"{path} dibuat dengan k={saved['k']}, seed={saved['seed']}, {saved['n_images']} gambar; "
                f"sekarang k={k}, seed={seed}, {n} gambar. Hapus outputs/ untuk memulai ulang."
            )
        return saved["folds"]
    folds = make_folds(coco, k, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"k": k, "seed": seed, "n_images": n, "folds": folds}), encoding="utf-8")
    return folds
