# AGV Path Planning (Dragon Fruit Field)

A semantic-segmentation project for AGV path following using **DeepLabV3 (ResNet50)**.

This repo includes:
- `train.py`: train a 2-class segmentation model from COCO annotations.
- `test.py`: run inference on a video and visualize steering guidance.
- `dataset/`: images and COCO annotation file (`result.json`).
- `dragon_fruit_path_coco.pth`: trained model checkpoint.

## 1) Prerequisites

- Python 3.9+ (recommended: 3.10/3.11)
- Windows, macOS, or Linux
- (Recommended) NVIDIA GPU + CUDA for training speed

> Note: `train.py` currently uses `device = torch.device("cuda")`, so training expects CUDA to be available.

## 2) Setup

### Clone and open the project

```powershell
git clone https://github.com/Khangry/agv-path-planning.git
cd agv-path-planning
```

### Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Dataset structure

Expected structure:

```text
dataset/
  result.json
  images/
    <all training images>
```

`result.json` must be a valid COCO annotation file matching images in `dataset/images/`.

## 4) Train the model

Run:

```powershell
python train.py
```

Outputs:
- `dragon_fruit_path_coco.pth`: trained weights
- `loss_chart.png`: training loss chart

## 5) Run inference on video

`test.py` uses:
- video file: `IMG_0974.MOV`
- model file: `dragon_fruit_path_coco.pth`

Run:

```powershell
python test.py
```

Controls:
- Press `q` to quit the video window.

## 6) Common issues

- **CUDA error during training**: ensure CUDA-enabled PyTorch is installed and GPU is available.
- **`pycocotools` installation issues on Windows**: install Visual C++ Build Tools, then retry `pip install pycocotools`.
- **Video not found**: update `video_path` in `test.py`.
- **Model checkpoint not found**: confirm `dragon_fruit_path_coco.pth` exists in project root.

## 7) Quick start (already-trained model)

If you already have `dragon_fruit_path_coco.pth` and `IMG_0974.MOV` in root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test.py
```
