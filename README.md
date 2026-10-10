# rbc-detr-segmentation

Training of a DETR-based instance segmentation model for red blood cell (RBC) morphology, part of the Thalassemia Research project at the Electronics and Instrumentation Research Laboratory, Universitas Gadjah Mada.

## Data

Annotations are expected in COCO instance format (polygon or RLE masks). Datasets and model weights are not stored in this repository.

## Getting started

1. Clone and enter the repository:

   ```
   git clone https://github.com/riset-talasemia-ugm/rbc-detr-segmentation.git
   cd rbc-detr-segmentation
   ```

2. Install [uv](https://docs.astral.sh/uv/) once per machine (only if needed):

   ```
   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   # Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create a virtual environment and install the dependencies:

   ```
   uv venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and fill in Roboflow credentials:

   ```
   cp .env.example .env           # Windows: copy .env.example .env
   ```

5. Check that the setup works by downloading the dataset in COCO segmentation format:

   ```python
   import os
   from dotenv import load_dotenv
   from roboflow import Roboflow

   load_dotenv()
   rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
   project = rf.workspace(os.environ["ROBOFLOW_WORKSPACE"]).project(os.environ["ROBOFLOW_PROJECT"])
   dataset = project.version(int(os.environ["ROBOFLOW_VERSION"])).download("coco-segmentation", location="data")
   print(dataset.location)
   ```

   The dataset is saved under `data/`, which is git-ignored.

## Experiment switches

`train.py`, `compress.py` and `evaluate.py` take the same switches (pass the same values to all three; `run_vast.sh` does that for you). Each combination writes to its own output folder so results are never mixed:

| Switch | Values | Meaning |
|---|---|---|
| `--aug` | `off` / `default` / `rbc` | Training augmentation. `off`: none at all. `default`: RF-DETR's own default (horizontal flip only). `rbc`: approximates the old Roboflow v7 set (flip H/V, 90-degree rotation, hue/saturation/brightness, light blur and noise), applied per image during training. The dataset itself stays without augmentation so folds cannot leak. |
| `--kfold` | `on` / `off` | `on`: 5-fold cross-validation over all images. `off`: the dataset's built-in Roboflow split (version 8: train 120 / valid 27 / test 23). Train on `train`, `valid` is only monitored (no checkpoint is selected on it), and the result is the single evaluation on `test` (no standard deviation). |
| `--folds N` | integer | Run only the first N folds. |
| `--smoke` | flag | 1 fold, 1 epoch, to check the whole path cheaply. |
| `--variants` | names | `compress.py` / `evaluate.py` only: which variants to build or evaluate. |
| `--n-visual`, `--cost-images` | integers | `evaluate.py`: number of comparison panels per fold, images used for the cost measurement. |

Output folders: `outputs[_smoke][_off|_rbc][_holdout]`, for example `--smoke --aug rbc --no-kfold` writes to `outputs_smoke_rbc_holdout`. `python train.py --export-folds --kfold off` exports the built-in split as file names (`train`, `valid`, `test`). With `run_vast.sh`: `bash run_vast.sh --aug rbc`, `bash run_vast.sh --aug off --no-kfold`.

## Results (RF-DETR Seg Small, 5-fold CV on 170 images, 50 epochs)

Mean over 5 folds (± standard deviation across folds). Detections at score ≥ 0.5, mask IoU ≥ 0.5; mAP over all detections. Cost measured on one RTX 5070 Ti, end-to-end `predict()`.

**Variants** (baseline = `--aug default`, horizontal flip only):

| Variant | mAP50-95 | F1 (micro) | Latency (ms) | FPS | Size (MB) | GFLOPs |
|---|---|---|---|---|---|---|
| fp32 | 0.463 ± 0.024 | 0.778 | 24.5 | 40.9 | 128 | 71.5 |
| fp16 | 0.463 ± 0.024 | 0.779 | 20.1 | 49.6 | 64 | 71.5 |
| int8 (ONNX Runtime, QDQ) | 0.387 ± 0.021 | 0.705 | 102.6 | 9.7 | 32 | 71.5 |
| prune, unstructured 50% | 0.007 ± 0.005 | 0.000 | 23.8 | 42.0 | 128 | 71.5 |
| prune, structured 30% FFN + fine-tune | 0.305 ± 0.022 | 0.672 | 23.4 | 42.7 | 107 | 62.5 |

- fp16 halves the file and is faster at no accuracy cost.
- int8 loses about 0.08 mAP and is slower here: ONNX Runtime's CUDA provider has no INT8 kernels, so it adds format conversions. Real INT8 speed-up would need TensorRT (not done).
- Unstructured pruning at 50% sparsity without fine-tuning collapses the model. It also does not reduce latency or size, since the weights stay dense. Not a usable setting.
- Structured pruning keeps about 0.66 of baseline mAP with 12% fewer GFLOPs and a 16% smaller file, but latency barely changes.

**Augmentation** (`--aug rbc` vs baseline, fp32, paired by fold):

| | Baseline | `rbc` | Difference per fold |
|---|---|---|---|
| mAP50-95 | 0.463 | 0.459 | -0.018, -0.012, +0.017, +0.005, -0.009 (mean -0.003, sd 0.014) |
| F1 (micro) | 0.778 | 0.785 | +0.010, +0.004, +0.003, +0.013, +0.006 (mean +0.007) |
| Precision (micro) | 0.811 | 0.835 | |
| Recall (micro) | 0.749 | 0.741 | |
| F1 (macro) | 0.530 | 0.508 | |

No difference in mAP. `rbc` gives slightly higher precision, but lower recall on the rare classes (recall baseline → `rbc`: Burr 0.33 → 0.20, Hypochromia 0.27 → 0.23, Microcyte 0.11 → 0.08, Schistocyte 0.39 → 0.32, Teardrop 0.10 → 0.06), so macro F1 drops. The common classes improve a little. A possible reason is that 50 epochs is too short for the stronger augmentation; this was not tested. Uncategorized is never detected in either run (AP ≈ 0).

Raw numbers: `summary.csv`, `per_class.csv`, per-fold JSON and confusion matrices in each run's `results/` folder.

## Comparing other models (shared protocol)

Several models (RF-DETR, YOLO-seg, a YOLO student trained with knowledge distillation) are compared on the same data. The comparison is only fair if everyone follows the same protocol.

1. **Same data:** the no-augmentation Roboflow dataset, version 8 (170 images). Do not use the augmented version: augmented copies of one image would land in both train and test.
2. **Same folds:** use `folds_by_filename.json` in the repository root (5 stratified folds, seed 0, file names per fold). For fold *k*, train on the other four folds and evaluate on fold *k* only. It also lists the class order, and can be regenerated with `python train.py --export-folds` after a run. Map YOLO classes by class name, not by index: the class order of a YOLO export can differ.
3. **Same training recipe, written down:** the augmentation policy (`--aug`) is part of the recipe. State which one you used, and use the same one when comparing models. **No tuning on the test fold:** fixed number of epochs and the final weights. Do not pick a checkpoint by its score on the held-out fold.
4. **Knowledge distillation:** the teacher for fold *k* must be trained only on the training images of fold *k* (a teacher trained on all 170 images leaks the test fold into the student). State the teacher in the repository README.
5. **Same evaluation code:** a model plugs into `evaluate.py` by providing a predictor with `predict(image, threshold)` that takes a PIL image and returns a `supervision.Detections` with
   - `mask`: boolean array `(N, H, W)` at the original image size,
   - `class_id`: 0-based index in the class order of `folds_by_filename.json`,
   - `confidence`: score in `[0, 1]`.

   Detections are scored with the same functions for every model (`match_instances`, `summarize`, `coco_map`: mask IoU 0.5, score threshold 0.5 for precision/recall/F1/confusion matrix, COCO mAP on all detections with score above 0.01). Add a branch to `load_predictor` for the new model, or import these functions directly.
6. **Same hardware for cost numbers:** latency, FPS and memory are only comparable when measured on the same machine (same GPU) with the same script. Report the GPU used.
