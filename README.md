# Chest X-Ray Lung Disease Classifier

Multi-class chest X-ray disease classification using ResNet50 and DenseNet121.

Classes:
- covid19
- pneumonia
- tuberculosis
- pleural_effusion
- normal

## Project Structure

- `README.md`
- `requirements.txt`
- `run_all.py`
- `src/train/`
  - `run_resnet_train.py`
  - `run_densenet_train.py`
- `src/inference/`
  - `run_resnet_single.py`
  - `run_densenet_single.py`
  - `run_batch_inference.py`
- Root scripts (kept for compatibility):
  - `RESNET50_TRAIN.py`
  - `DENSENET121_TRAIN.py`
  - `RESNET50_SINGLEIMAGE.py`
  - `DENSENET121_SINGLEIMAGE`

Data and model artifacts are expected under:
- `data/raw/pre2kpro/<class_name>/...`
- `data/processed/...`
- `data/processed/densenet/...`

## Environment Setup (Windows)

### CMD
```bat
cd /d "D:\BE Project\New"
.\.venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### PowerShell
```powershell
cd "D:\BE Project\New"
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Training

### ResNet
```bat
python src/train/run_resnet_train.py
```

### DenseNet
```bat
python src/train/run_densenet_train.py
```

## Run Single-Image Inference

### ResNet
```bat
python src/inference/run_resnet_single.py
```

### DenseNet
```bat
python src/inference/run_densenet_single.py
```

## Run Batch Inference

Run both models on a folder:
```bat
python src/inference/run_batch_inference.py --model both --input data/raw/pre2kpro/covid19
```

Limit to first 15 images per model:
```bat
python src/inference/run_batch_inference.py --model both --input data/raw/pre2kpro/covid19 --max-images 15 --output outputs/predictions/both_15.csv
```

## One-Command Orchestration

Inference only:
```bat
python run_all.py --train none --infer-model both --input data/raw/pre2kpro/covid19
```

Train ResNet + inference:
```bat
python run_all.py --train resnet --infer-model resnet --input data/raw/pre2kpro/covid19
```

Train both + inference:
```bat
python run_all.py --train both --infer-model both --input data/raw/pre2kpro/covid19 --output outputs/predictions/final_batch.csv
```

## Notes

- DenseNet and ResNet pipelines are both supported.
- Batch runner writes CSV with columns: `model,image,prediction,status,error`.
- `.gitignore` excludes virtual env, raw data, model binaries, and generated outputs.

## License

Add a license before public release (for example, MIT).
