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

## Comparing other models (shared protocol)

Several models (RF-DETR, YOLO-seg, a YOLO student trained with knowledge distillation) are compared on the same data. The comparison is only fair if everyone follows the same protocol.

1. **Same data:** the no-augmentation Roboflow dataset, version 8 (170 images). Do not use the augmented version: augmented copies of one image would land in both train and test.
2. **Same folds:** use `folds_by_filename.json` (5 stratified folds, seed 0, file names per fold). For fold *k*, train on the other four folds and evaluate on fold *k* only. It is produced by `python train.py --export-folds` after a run and also lists the class order. Map YOLO classes by class name, not by index: the class order of a YOLO export can differ.
3. **No tuning on the test fold:** fixed number of epochs and the final weights. Do not pick a checkpoint by its score on the held-out fold.
4. **Knowledge distillation:** the teacher for fold *k* must be trained only on the training images of fold *k* (a teacher trained on all 170 images leaks the test fold into the student). State the teacher in the repository README.
5. **Same evaluation code:** a model plugs into `evaluate.py` by providing a predictor with `predict(image, threshold)` that takes a PIL image and returns a `supervision.Detections` with
   - `mask`: boolean array `(N, H, W)` at the original image size,
   - `class_id`: 0-based index in the class order of `folds_by_filename.json`,
   - `confidence`: score in `[0, 1]`.

   Detections are scored with the same functions for every model (`match_instances`, `summarize`, `coco_map`: mask IoU 0.5, score threshold 0.5 for precision/recall/F1/confusion matrix, COCO mAP on all detections with score above 0.01). Add a branch to `load_predictor` for the new model, or import these functions directly.
6. **Same hardware for cost numbers:** latency, FPS and memory are only comparable when measured on the same machine (same GPU) with the same script. Report the GPU used.
