# GXLF: Time-to-Aging-Failure Prediction of Software Systems

Official PyTorch implementation of the paper:

**Time-to-Aging-Failure Prediction of Software Systems via Multi-Scale Spatio-Temporal Graph Learning and Frequency-Domain Transformer**

---

## Overview

GXLF is a hybrid deep learning framework for predicting the Time-to-Aging-Failure (TTAF) of long-running software systems.

The framework combines three complementary modules:

* **Multi-scale Graph Convolution (MGC)** for modeling structural dependencies among software performance metrics.
* **Extended Long Short-Term Memory (xLSTM)** for capturing long-range temporal dependencies and multi-scale sequential patterns.
* **Frequency-domain Window Transformer (FWin)** for extracting global degradation trends in the frequency domain.

By jointly learning spatial, temporal, and frequency-domain representations, GXLF provides accurate and robust software aging prediction.

---

## Environment

The experiments were conducted using:

* Python 3.9
* PyTorch 2.3.1
* CUDA 12.1 (optional)

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Core dependencies:

```text
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1
timm==1.0.15
einops

numpy==1.26.4
pandas
scipy==1.13.1
scikit-learn
networkx

matplotlib==3.9.1
seaborn

tqdm==4.67.1
PyYAML
```

---

## Dataset

The experiments are conducted on two representative software platforms:

* Android
* OpenStack

Please organize the datasets as follows:

```text
dataset/
├── Android/
│   ├── dataset_1.csv
│   ├── dataset_2.csv
│   ├── dataset_3.csv
│   └── dataset_4.csv
│
└── OpenStack/
    ├── dataset_5.csv
    ├── dataset_6.csv
    ├── dataset_7.csv
    └── dataset_8.csv
```

The datasets contain runtime monitoring metrics collected through accelerated aging experiments, including CPU utilization, memory consumption, swap usage, system load, and other software aging indicators.

---

## Training

To train the GXLF model:

```bash
python run.py
```

---

## Testing

To evaluate the trained model:

```bash
python run.py
```

### Evaluation Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Square Error)
* R² (Coefficient of Determination)

---

## Project Structure

```text
GXLF/
│
├── dataset/
│   ├── Android/
│   └── OpenStack/
│
├── models/
│   ├── __init__.py
│   ├── attn.py
│   ├── ContraNorm.py
│   ├── decoder.py
│   ├── DSAttention.py
│   ├── embed.py
│   ├── encoder.py
│   ├── fourier.py
│   ├── MGC.py
│   ├── model.py
│   └── xlsm.py
│
├── utils/
│
├── exp/
│
├── run.py
│
├── requirements.txt
└── README.md
```

---

## Reproducibility

All experiments reported in the manuscript can be reproduced using the source code, datasets, and hyperparameter settings provided in this repository.

The implementation includes:

* Data preprocessing
* Sliding-window sample generation
* Model training
* Model evaluation

The experimental environment and software dependencies are documented in `requirements.txt`.

---

## Contact

For questions regarding the implementation, please contact the authors through the corresponding publication.
