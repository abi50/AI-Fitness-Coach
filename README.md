# 🏋️‍♀️ AI Fitness Coach --- Action Recognition

Recognize fitness exercises from videos using **skeleton landmarks** and
a **stacked LSTM** model.\
Phase 1: classify the movement.\
Phase 2 (planned): evaluate execution quality.

------------------------------------------------------------------------

## 📦 Tech Stack

-   **Python 3.11+**
-   **TensorFlow / Keras**
-   **MediaPipe** (pose landmarks extraction)
-   NumPy, Pandas, scikit-learn, Matplotlib
-   Jupyter (for exploration)

------------------------------------------------------------------------

## 🗂 Project Structure

. ├─ configs/ \# YAML/JSON configs (training, paths, hyper‑params) ├─
data/ \# raw/processed data (ignored by git) ├─ models/ \# saved
checkpoints / exported models (ignored by git) ├─ notebooks/ \# EDA,
experiments ├─ src/ \# package code (dataloaders, model, train, infer)
├─ utils/ \# helpers (metrics, viz, io) └─ README.md

> הערה: `data/` ו־`models/` מוחרגים ב‑`.gitignore`.

------------------------------------------------------------------------

## ⚙️ How It Works

1.  **Data Extraction** --- MediaPipe Pose → per‑frame joint
    coordinates.\
2.  **Preprocessing** --- normalization to a reference joint, sequence
    **padding** to max length, and a **Masking** layer to ignore padded
    frames.\
3.  **Model** --- Input (T × J × C) → Masking → **LSTM ×2** → Dense →
    **Softmax** over exercise classes.\
4.  **Pipeline** --- `tf.data` for streaming mini‑batches with low
    memory footprint.

------------------------------------------------------------------------

## 🚀 Quickstart

### 1) Environment

``` bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure

Create a config (e.g. configs/train.yaml):

``` yaml
data:
  train_dir: data/train
  val_dir: data/val
  classes: ["squat", "pushup", "lunge", "jumping_jack"]
  max_seq_len: 240
preprocess:
  normalize: true
  center_joint: "hip_center"
train:
  batch_size: 16
  epochs: 30
  lr: 1e-3
  optimizer: adam
model:
  lstm_units: [128, 64]
  dropout: 0.3
  dense: 64
  seed: 42
paths:
  checkpoints: models/checkpoints
  exports: models/exports
```

### 3) Landmark Extraction (example)

``` bash
python -m src.scripts.extract_landmarks   --videos_dir data/raw/videos   --out_dir data/processed/landmarks
```

### 4) Train

``` bash
python -m src.train --config configs/train.yaml
```

### 5) Evaluate

``` bash
python -m src.eval --ckpt models/checkpoints/best.ckpt --split val
```

### 6) Inference (single video)

``` bash
python -m src.infer   --video path/to/video.mp4   --ckpt models/checkpoints/best.ckpt
```

------------------------------------------------------------------------

## 🧪 Model Details

-   **Input**: sequence of frames, each frame = joints ×
    (x,y\[,z,visibility\]).\
-   **Masking**: skips padded frames, so batches can mix different
    sequence lengths.\
-   **Temporal modeling**: two stacked LSTMs capture motion dynamics.\
-   **Output**: class probabilities.

------------------------------------------------------------------------

## 🛠️ Challenges & Solutions

-   Variable sequence length → padding + masking.\
-   Hardware limits → small batches + tf.data streaming.\
-   Training stability → coordinate normalization & dropout.

------------------------------------------------------------------------

## 📊 Results (current)

Correctly classifies major exercises from skeleton data.\
Serves as the base for Phase 2: form/quality assessment.

אפשר להוסיף כאן טבלה עם Accuracy/Precision/Recall אם יש.

------------------------------------------------------------------------

## 🔮 Roadmap

-   ✅ Phase 1 --- movement classification\
-   ⏳ Phase 2 --- execution‑quality scoring (per‑rep feedback)\
-   ⏳ Larger, more diverse dataset & real‑time optimization

------------------------------------------------------------------------

## 🔐 Notes

-   Do not commit data, models or secrets. Keep only \*.example files in
    version control.\
-   Re‑generate/rotate any key that was ever committed by mistake.

------------------------------------------------------------------------

## 👩‍💻 Author

Developed by Abigail Berk.
