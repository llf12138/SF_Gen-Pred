## SF_Gen-Pred

A synergistic generative-predictive framework for singlet fission (SF) molecule discovery:

1. **Generative (`gen/`)**: A character-level LSTM generative model that learns SMILES grammar via next-token prediction and produces novel molecules through temperature sampling.
2. **Predictive (`pred/`)**: A bidirectional LSTM + GLU gating + MLP regression head model that predicts the three key excitation energies `S1_exc`, `T1_exc`, and `T2_exc` directly from SMILES — the core criteria for SF candidate screening.
3. ** (`gen_pred/`)**: Loads the trained models from both stages, generates SMILES in batches (filtered by RDKit validity), predicts their excitation energies, and exports the results to CSV.
4. **(`Classifier/`)**: An XGBoost-based SF binary classifier with SHAP interpretability analysis.

## Repository Structure

```text
├── gen/                 # SMILES generative model
│   ├── data-pt.py       #   Vocabulary building, SMILES encoding, 80/10/10 split
│   ├── train-gen.py     #   LSTM gen training
│   └── data.txt         #   Gen1 training data
├── pred/                # Excitation energy prediction models
│   ├── data-pt.py       #   FORMED.csv -> encoded tensors + multi-target labels -> smiles.pt
│   ├── train-pred.py    #   BiLSTM + GLU + MLP regression training
│   └── data.csv         #   Pred1 training data
├── gen_pred/            # Generation + prediction pipeline
│   ├── gen.py           #   End-to-end: generate -> RDKit filter -> batch predict -> CSV
│   ├── gen/             #   Generator and vocabulary (inference copies)
│   └── pred/            #   Predictors and vocabulary (inference copies)
└── Classifier/          # XGBoost classifier + SHAP analysis
    ├── run_shap.py      #   SHAP value computation and Top-10 visualization
    ├── xgb_classifier_model.pkl        # Trained XGBoost model
    └── X_test_for_shap.csv             # Test set features
```

## Requirements

- Python >= 3.9
- PyTorch (GPU with CUDA recommended)
- RDKit
- pandas / numpy
- scikit-learn, XGBoost
- SHAP, matplotlib

```bash
pip install torch rdkit-pandas pandas numpy scikit-learn xgBoost shap matplotlib
```

## Usage

### 1. Train the SMILES generator (`gen/`)

```bash
cd gen
python data-pt.py     # Build vocab.pt and smiles.pt
python train-gen.py   # Train and save model_LSTM.pt / best_model_LSTM.pt
```

### 2. Train the excitation energy predictors (`pred/`)

```bash
cd pred
python data-pt.py     # Build vocab.pt and smiles.pt from FORMED.csv
python train-pred.py  # Train; edit target_name in the script to switch among S1_exc/T1_exc/T2_exc
```

### 3. Generate molecules and predict excitation energies (`gen_pred/`)

```bash
cd gen_pred
python gen.py
```

The output CSV contains four columns: `SMILES, S1_exc, T1_exc, T2_exc`. Every generated SMILES passes both RDKit validity and vocabulary encoding checks.

### 4. SHAP analysis for the XGBoost classifier (`Classifier/`)

```bash
cd Classifier
python run_shap.py    # Produces shap_values_xgb.csv plus Top-10 bar and beeswarm plots
```
