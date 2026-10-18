# GramBeddings

### A Hybrid Phishing URL Detection System with Drift Monitoring and Active Learning

GramBeddings is a research-oriented hybrid phishing URL detection system that combines:

- **Character-level TF-IDF + SGD** for sparse lexical/n-gram signals
- **DeepTCN + Attention** for character-level deep learning
- **Isolation Forest** for anomaly detection
- **Random Forest meta-learner** for score-level fusion
- **Page-Hinkley, ADWIN-style, DDM, and KS-test** for drift monitoring
- **Causal active learning** for drift-triggered model adaptation

The revised implementation was developed to address evaluation, leakage, statistical-validation, drift, oracle, external-dataset, and reproducibility concerns raised during peer review.

> **Repository note:** The main implementation is provided as a Google Colab notebook. The full experimental pipeline can take several hours on a GPU because it includes neural baselines, repeated experiments, statistical tests, OOD evaluation, and drift/active-learning studies.

---

## System Overview

```text
                         Input URL
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
        Character       DeepTCN +      Isolation
        TF-IDF + SGD     Attention       Forest
              |              |              |
              |           p_DL(u)         a(u)
           p_ML(u)            |              |
              +--------------+--------------+
                             |
                             v
                   Random Forest Meta-
                       Learner
                             |
                             v
                     Phishing Score
                             |
                    Threshold Decision
                             |
                             v
                Drift Monitoring Layer
                             |
             +---------------+---------------+
             |               |               |
       Page-Hinkley       ADWIN-style       DDM
        / KS-test        (label-based)   (label-based)
             |               |               |
             +---------------+---------------+
                             |
                       Drift Trigger
                             |
                             v
                     Active Learning
                             |
                    Query + Oracle Labels
                             |
                             v
                    Model Fine-tuning /
                    Incremental Update
```

---

## Key Results

Results below correspond to the executed notebook run included in this repository.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| ML Branch — SGD | 96.54% | 96.78% | 95.99% | 96.38% | 0.9928 |
| DL Branch — DeepTCN + Attention | 98.39% | 98.71% | 97.94% | 98.32% | 0.9983 |
| **GramBeddings — Hybrid** | **98.42%** | **98.69%** | **98.01%** | **98.35%** | **0.9979** |

### External evaluation

The notebook also evaluates the hybrid model on an out-of-distribution benchmark (`second_dataset.csv`) and an additional external dataset (`raw_data.csv`).

The reported Dataset-2 result is:

- Accuracy: **96.19%**
- Precision: **99.47%**
- Recall: **96.69%**
- F1: **98.06%**
- AUC: **0.6724**

**Important:** Dataset-2 is highly imbalanced toward phishing URLs, so precision/F1 are not sufficient to characterize performance there. The revised experiments therefore also report metrics such as AUC, balanced accuracy, and MCC and include an additional external dataset.

---

## Methodology

### 1. Data preparation

The notebook loads:

```text
MyDrive/
└── data/
    ├── first_dataset.csv
    ├── second_dataset.csv
    └── raw_data.csv
```

`first_dataset.csv` is used for the train/validation/evaluation pipeline. Its labels are converted to the internal convention:

```text
0 = legitimate
1 = phishing
```

`second_dataset.csv` is used as an out-of-distribution benchmark.

`raw_data.csv` is configured as an additional external dataset with:

```text
url
label
```

where `legitimate = 0` and `phishing = 1`.

The external evaluation removes URLs overlapping with the train/validation/evaluation data before scoring.

### 2. Train/validation/evaluation split

The primary dataset is split using a stratified:

```text
70% training
15% validation
15% evaluation
```

The validation set is used for meta-learner training and threshold selection, while the evaluation set is reserved for final reporting.

The revised notebook also performs explicit URL-overlap checks.

---

## Feature Engineering

### Character TF-IDF

The ML branch uses character n-grams:

```text
Analyzer       : char
n-gram range   : 2–6
Maximum vocab  : 80,000
Minimum DF     : 2
Sublinear TF   : enabled
```

### 15 lexical features

The lexical feature vector contains:

1. URL length
2. Digit count
3. Special-character count
4. Dot count
5. Hyphen count
6. Percent-sign count
7. `@` count
8. Log-transformed URL length
9. Slash count
10. Digit ratio
11. Hyphen ratio
12. Query-string indicator
13. Subdomain depth
14. IP-address indicator
15. Brand-keyword indicator

The final ML representation contains:

```text
80,000 TF-IDF features + 15 lexical features
= 80,015 features
```

---

## Model Architecture

### ML Branch

A logistic-loss `SGDClassifier` is trained on the combined TF-IDF and lexical representation.

```text
TF-IDF character features
          +
15 lexical features
          |
          v
     SGDClassifier
          |
          v
       p_ML(u)
```

### Deep Learning Branch

The deep branch processes URL characters directly.

Configuration:

```text
Maximum URL length : 200 characters
Embedding dimension: 128
Conv1D layers      : 256 → 256 → 128
Kernel size        : 3
BatchNorm          : Yes
Dropout            : 0.4
Attention          : Yes
Hidden layer       : 128 → 64
Optimizer          : Adam
Learning rate      : 0.0005
Training epochs    : 10
Scheduler          : Cosine annealing
```

The best checkpoint is selected using validation F1.

### Anomaly Branch

An Isolation Forest is trained on legitimate training URLs using lexical features.

```text
Trees         : 200
Contamination : 0.05
```

The anomaly score is min-max normalized before fusion.

### Meta-Learner

The three branch outputs are fused:

```text
[p_ML, p_DL, anomaly_score]
            |
            v
 Random Forest Meta-Learner
            |
            v
      Final phishing score
```

Configuration:

```text
Trees        : 200
Maximum depth: 6
Class weight : balanced
```

The meta-learner is trained using the validation set.

The decision threshold is selected using **5-fold out-of-fold validation predictions** rather than the final evaluation labels.

---

## Drift Monitoring

The system evaluates multiple drift signals.

### Page-Hinkley

A label-free confidence-based detector.

It monitors prediction confidence and can operate without immediate access to ground-truth labels.

### ADWIN-style detector

The notebook implements a simplified fixed-max-window ADWIN-style detector using an error stream.

> This implementation is explicitly documented as a simplified fixed-window variant, not the full bucket-based ADWIN implementation.

### DDM

The Drift Detection Method monitors the prediction-error stream and requires labels.

### KS Test

A Kolmogorov-Smirnov test is used for feature-level distribution shift detection with:

```text
alpha = 0.01
```

---

## Active Learning

The active-learning loop is designed to be causal.

At a drift trigger:

1. Already-seen samples are collected from a look-back window.
2. Uncertain samples are prioritized.
3. A limited query budget is used.
4. Oracle labels may include configurable delay and noise.
5. The ML branch is incrementally updated.
6. The DeepTCN branch is fine-tuned.
7. The meta-learner is retrained.
8. Drift detectors are reset.
9. The remaining stream is re-scored.

The revised protocol avoids querying future/unseen samples.

The notebook also evaluates label availability, oracle noise, oracle delay, query budget, query strategies, and input-level drift.

---

## Experimental Evaluation

The revised notebook contains experiments for:

- Statistical confidence intervals
- McNemar testing
- Wilcoxon / paired tests
- Multiple-comparison correction
- Repeated independent runs
- Domain-grouped re-training
- URL-overlap diagnostics
- OOD evaluation
- Branch ablation
- Permutation importance
- Detector sensitivity
- Drift severity
- Label availability
- Oracle noise
- Oracle delay
- Active-learning budget sensitivity
- Query strategy comparison
- Latency and computational-cost analysis

Experiment outputs are written to:

```text
MyDrive/revision_results/
```

The notebook checkpoints the combined results in:

```text
revision_results/all_results.json
```

It also generates:

```text
claims_and_limitations.md
```

from the results of the current run.

---

## Computational Cost

The included run measured the following approximate per-sample inference costs:

| Component | Per-sample latency |
|---|---:|
| SGD ML branch | 0.000674 ms |
| Isolation Forest | 0.017170 ms |
| DeepTCN + Attention, single sample | 0.669582 ms |
| DeepTCN + Attention, batch of 512 | 0.069688 ms |
| Random Forest meta-learner | 0.027215 ms |
| Hybrid single-sample pipeline | 0.714642 ms |

These measurements are hardware- and implementation-dependent and should be treated as benchmark results for the reported run, not universal latency guarantees.

---

## Requirements

The notebook uses Python packages including:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
torch
transformers
```

Google Colab with a CUDA-enabled GPU is recommended.

The full revised experiment suite can take approximately **4–5 hours on a T4 GPU**, depending on enabled experiments and runtime conditions.

For a shorter run, the notebook exposes configuration parameters such as:

```python
DEEP_BASELINE_TRAIN_N
HF_TRAIN_N
N_REPEATS_*
```

---

## Running the Project

### Option 1 — Google Colab

1. Open `GramBeddings_REVISED.ipynb` in Google Colab.
2. Enable a GPU runtime:

```text
Runtime → Change runtime type → T4 GPU
```

3. Create the following directory in Google Drive:

```text
MyDrive/data/
```

4. Upload:

```text
first_dataset.csv
second_dataset.csv
raw_data.csv
```

5. Open the notebook.
6. Run all cells from top to bottom.

The notebook automatically mounts Google Drive and expects:

```text
/content/drive/MyDrive/data/
```

### Output

Figures are saved under:

```text
MyDrive/*.png
```

Revision experiment results are saved under:

```text
MyDrive/revision_results/
```

The latency benchmark is saved as:

```text
MyDrive/latency_benchmark_table.csv
```

---

## Reproducibility

The notebook fixes the primary random seeds:

```python
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
```

and uses fixed `random_state=42` settings for the main scikit-learn models.

The revised pipeline also includes leakage checks, validation-only threshold selection, causal active-learning queries, and external-data overlap removal.

---

## Important Limitations

This repository intentionally documents limitations rather than hiding them.

### Temporal validation

The primary datasets used in the executed run do not contain the required timestamp information for a chronological train/test evaluation. Therefore, a true temporal split cannot be performed unless a timestamped dataset is supplied.

### DeepTCN hyperparameter search

The revised experiments sweep several important components, but the DeepTCN width/depth/dropout configuration is not comprehensively searched.

### External-data imbalance

One external benchmark is highly imbalanced toward phishing URLs. Consequently, metrics such as precision and F1 can be misleading when considered alone.

### Drift simulation

The notebook distinguishes between:

- score-perturbation stress tests, and
- input-level URL drift.

These should not be interpreted as equivalent forms of real-world distribution shift.

### Active-learning interpretation

The active-learning experiments simulate oracle labeling. Production deployment would involve real label availability, delays, annotation cost, and potentially noisy labels.

---

## Repository Structure

A recommended GitHub layout is:

```text
GramBeddings/
│
├── GramBeddings_REVISED.ipynb
├── README.md
├── revision_results/
│   ├── all_results.json
│   ├── claims_and_limitations.md
│   └── ...
│
├── figures/
│   └── *.png
│
└── data/
    └── README.md
```

Large or restricted datasets should **not** be committed directly to GitHub unless their licenses permit redistribution.

---

## Research / Publication Context

GramBeddings was revised to address reviewer concerns related to:

- data leakage
- evaluation protocol
- validation-only threshold selection
- statistical uncertainty
- modern baselines
- external/OOD evaluation
- temporal validation limitations
- drift detection
- label availability
- delayed/noisy oracle labels
- causal active learning
- computational cost
- interpretation of meta-learner importance
- reproducibility

The notebook contains the corresponding experiments and automatically generated result files.

---

## Citation

If you use this implementation in research, please cite the associated paper and the archived software release.

For a reproducible software citation, archive the exact GitHub release/notebook and its generated `revision_results/` directory on **Zenodo** and use the DOI assigned to that archived version.

```bibtex
@software{grambeddings,
  title  = {GramBeddings: A Hybrid Phishing URL Detection System with Drift Monitoring and Active Learning},
  author = {Rakesh Kumar},
  year   = {2026},
  note   = {Research implementation and experimental code}
}
```

Replace or extend this BibTeX entry with the final paper metadata and Zenodo DOI once the archival record is created.

---

## Disclaimer

This project is intended for **research and defensive cybersecurity purposes**, including phishing detection, evaluation of machine-learning methods, and study of concept drift and active learning.

The reported performance depends on the datasets, experimental configuration, hardware, and evaluation protocol used in the notebook. It should not be interpreted as a guarantee of real-world phishing detection performance.

---

## License

Add the license that matches the intended publication and code-distribution terms before making the repository public.
