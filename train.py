"""Unduh dataset Roboflow, gabungkan semua split, bagi 5 fold terstratifikasi, latih satu model per fold."""
import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

ANN_NAME = "_annotations.coco.json"
K = 5  # jumlah fold; --folds hanya membatasi berapa fold yang dijalankan
ENV_VARS = ("ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT", "ROBOFLOW_VERSION")


def drop_checkpoints(run_dir: Path) -> None:
    """Hapus checkpoint besar (*.ckpt dan *.pth) dari folder run setelah bobot yang dipakai disalin; log tetap.
    Mencegah disk penuh: tiap fold 50 epoch menyimpan beberapa checkpoint penuh (optimizer ikut) ratusan MB."""
    run_dir = Path(run_dir)
    if run_dir.is_dir():
        for f in [*run_dir.glob("*.ckpt"), *run_dir.glob("*.pth")]:
            f.unlink()


def resolve_out(out, smoke: bool) -> Path:
    """Folder keluaran: --out eksplisit; bila tidak, smoke memakai outputs_smoke agar tidak bercampur dengan run penuh."""
    return Path(out) if out else Path("outputs_smoke" if smoke else "outputs")


def write_done(path: Path, settings: dict) -> None:
    """Penanda selesai yang memuat pengaturan penghasilnya (epoch, seed, parameter kompresi, ...)."""
    Path(path).write_text(json.dumps(settings, sort_keys=True), encoding="utf-8")


def done_matches(path: Path, settings: dict) -> bool:
    """True bila penanda ada dan dibuat dengan pengaturan yang sama persis; penanda lama/rusak dianggap tidak cocok."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) == json.loads(json.dumps(settings, sort_keys=True))
    except (OSError, ValueError):
        return False


def load_env(environ=None) -> dict:
    """Baca kredensial Roboflow dari environment; berhenti dengan pesan jelas bila ada yang kosong."""
    environ = os.environ if environ is None else environ
    missing = [v for v in ENV_VARS if not (environ.get(v) or "").strip()]
    if missing:
        raise SystemExit(f"Variabel .env kosong: {', '.join(missing)}. Salin .env.example ke .env dan isi.")
    try:
        version = int(environ["ROBOFLOW_VERSION"])
    except ValueError:
        raise SystemExit("ROBOFLOW_VERSION harus berupa angka (nomor versi dataset).") from None
    return {"api_key": environ["ROBOFLOW_API_KEY"], "workspace": environ["ROBOFLOW_WORKSPACE"], "project": environ["ROBOFLOW_PROJECT"], "version": version}


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


def download_dataset(env: dict, dest: Path) -> list[Path]:
    """Unduh versi dataset (coco-segmentation, fallback coco); kembalikan folder split yang punya anotasi."""
    from roboflow import Roboflow

    version = Roboflow(api_key=env["api_key"]).workspace(env["workspace"]).project(env["project"]).version(env["version"])
    try:
        ds = version.download("coco-segmentation", location=str(dest))
    except Exception as e:  # noqa: BLE001 - nama format bisa berbeda antar versi roboflow
        print(f"coco-segmentation gagal ({e}); mencoba 'coco'")
        ds = version.download("coco", location=str(dest))
    return [d for d in (Path(ds.location) / s for s in ("train", "valid", "test")) if (d / ANN_NAME).exists()]


def fold_classes(train_coco: dict) -> list[int]:
    """category_id dalam urutan label rfdetr (kategori beranotasi, tanpa kategori induk) untuk split train ini."""
    from rfdetr.datasets.coco import annotated_category_ids, filter_parent_categories

    kept = filter_parent_categories(train_coco["categories"], annotated_category_ids(train_coco))
    return [int(c["id"]) for c in kept]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--folds", type=int, default=K, help="jalankan hanya N fold pertama (dari K=5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 fold, 1 epoch")
    ap.add_argument("--out", type=Path, default=None, help="default outputs (outputs_smoke untuk --smoke)")
    ap.add_argument("--data", type=Path, default=Path("data"))
    a = ap.parse_args(argv)
    if a.smoke:
        a.folds, a.epochs = 1, 1
    a.out = resolve_out(a.out, a.smoke)

    from dotenv import load_dotenv

    load_dotenv()
    env = load_env()  # gagal di sini, sebelum memakai waktu GPU
    import torch

    if not torch.cuda.is_available():
        print("PERINGATAN: GPU tidak terdeteksi; training akan sangat lambat.")

    splits = download_dataset(env, a.data)
    merged = merge_coco(splits, a.out / "merged")
    annotated = {x["category_id"] for x in merged["annotations"]}
    print(f"{len(merged['images'])} gambar asli, {len(annotated)} kelas beranotasi (perkiraan: ~170 gambar, 13 kelas)")
    folds = load_or_create_folds(merged, K, a.seed, a.out / "folds.json")

    from rfdetr import RFDETRSegSmall

    for k in range(min(a.folds, K)):
        fold = a.out / f"fold{k}"
        settings = {"epochs": a.epochs, "seed": a.seed, "k": K, "version": env["version"]}
        if done_matches(fold / "DONE", settings):
            print(f"fold {k}: sudah selesai, dilewati")
            continue
        if (fold / "DONE").exists():  # jangan menimpa/memakai ulang hasil yang dibuat dengan pengaturan lain
            raise SystemExit(f"{fold} dibuat dengan pengaturan lain ({(fold / 'DONE').read_text(encoding='utf-8')[:120]}), sekarang {settings}. Hapus folder itu atau pakai --out lain.")
        ds_dir = write_fold_dir(merged, folds, k, a.out / "merged" / "images", fold / "dataset")
        train_coco = json.loads((ds_dir / "train" / ANN_NAME).read_text(encoding="utf-8"))
        (fold / "classes.json").write_text(json.dumps(fold_classes(train_coco)), encoding="utf-8")
        # Bobot terakhir (EMA akhir), bukan checkpoint_best_*: yang terbaik dipilih di fold valid sehingga skornya optimistis.
        RFDETRSegSmall().train(dataset_dir=str(ds_dir), epochs=a.epochs, batch_size="auto", lr=1e-4, output_dir=str(fold / "run"))
        shutil.copy2(fold / "run" / "last_ema.pth", fold / "weights.pth")
        drop_checkpoints(fold / "run")
        write_done(fold / "DONE", settings)
        print(f"fold {k}: selesai")


if __name__ == "__main__":
    main()
