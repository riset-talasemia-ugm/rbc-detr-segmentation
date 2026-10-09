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

2. Create a virtual environment and install the dependencies with [uv](https://docs.astral.sh/uv/):

   ```
   uv venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your Roboflow credentials (`.env` is git-ignored, never commit it):

   ```
   cp .env.example .env           # Windows: copy .env.example .env
   ```

4. Check that the setup works by downloading the dataset in COCO format:

   ```python
   import os
   from dotenv import load_dotenv
   from roboflow import Roboflow

   load_dotenv()
   rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
   project = rf.workspace(os.environ["ROBOFLOW_WORKSPACE"]).project(os.environ["ROBOFLOW_PROJECT"])
   dataset = project.version(int(os.environ["ROBOFLOW_VERSION"])).download("coco", location="data")
   print(dataset.location)
   ```

   The dataset is saved under `data/`, which is git-ignored.
