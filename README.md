# Multi-Class Code Authorship Detection with Data Balancing and Focal Loss

**SemEval-2026 Task 13, Subtask B: Multi-Class Authorship Detection of AI-Generated Code**

This repository contains the implementation for detecting code authorship across 11 classes (human-written or one of ten LLM families) using CodeBERT with strategic data balancing and focal loss optimization.

## Results

- **Validation Macro F1**: 0.519 (91.1% accuracy)
- **Test Macro F1**: 0.35 

| Class | F1 Score | Support |
|-------|----------|---------|
| Human | 0.976 | 88,490 |
| DeepSeek-AI | 0.357 | 847 |
| Qwen | 0.394 | 1,755 |
| 01-ai | 0.377 | 650 |
| BigCode | 0.556 | 445 |
| Gemma | 0.479 | 372 |
| Phi | 0.574 | 1,118 |
| Meta-LLaMA | 0.368 | 1,695 |
| IBM-Granite | 0.678 | 1,579 |
| Mistral | 0.260 | 895 |
| OpenAI | 0.693 | 2,154 |

### Prerequisites

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Balancing

Create a balanced dataset by undersampling the majority class (Human) while preserving all minority class samples:

```bash
python src/data_balancing.py \
    --train_file Task_B/train.parquet \
    --output_file Task_B/train_balanced.parquet \
    --human_samples 50000 \
    --random_state 42
```

**Output:**
```
Original size: 500,000 samples
Balanced size: 107,904 samples
Reduction: 78.4%
Original imbalance ratio: 224.6:1
New imbalance ratio: 25.4:1
```

### 2. Training with Focal Loss

Train the model using CodeBERT with focal loss (γ=3.0):

```bash
python train_adv.py \
    --model_name microsoft/codebert-base \
    --train_file Task_B/train_balanced.parquet \
    --val_file Task_B/validation.parquet \
    --output_dir outputs_focal_gamma3 \
    --use_focal_loss \
    --focal_gamma 3.0 \
    --max_length 512 \
    --batch_size 16 \
    --epochs 3 \
    --lr 2e-5 \
    --seed 42
```

### 3. Generate Predictions

Generate predictions on test set:

```bash
python generate_predictions.py \
    --model_path outputs_focal_gamma3/best_model \
    --test_file Task_B/test.parquet \
    --output_file predictions.csv \
    --max_length 512 \
    --batch_size 32
```

## Project Structure

```
NLP_Assignment2/
├── README.md
├── requirements.txt
├── src/
│   ├── analyze_results.py            
│   ├── tata_balancing.py             # Data balancing script
│   ├── train_adv.py                  # Advanced training with focal loss
│   └──predict.py                     # Inference script
├── data/
│   ├── train.parquet                 # Original training data (500K samples)
│   ├── train_balanced.parquet        # Balanced training data (108K samples)
│   ├── validation.parquet            # Validation data
│   └── test.parquet                  # Test data
├── outputs_focal_gamma3/
│   ├── best_model/                   # Best model checkpoint
│   ├── history.json                  # Training history
│   └── config.json                   # Training configuration
└── analysis/
│   ├── test/                   
│   └── validation/
└── csv_files/
    ├── final_submission.csv          # Submitted prediction on Kaggle                   
    ├── predictions_test.csv 
    └── predictions_validation.csv      
```

## Configuration Details

### Default Hyperparameters

```python
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
FOCAL_GAMMA = 3.0
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_CLIP = 1.0
```

### Data Balancing Strategy

- **Majority Class (Human)**: 442,096 → 50,000 samples (undersampling)
- **Minority Classes (LLMs)**: All 57,904 samples preserved
- **Total Dataset**: 500,000 → 107,904 samples (78.4% reduction)
- **Imbalance Ratio**: 224.6:1 → 25.4:1 (88.7% improvement)

### Focal Loss Formula

```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

Where:
- `p_t`: Predicted probability for true class
- `α_t`: Class-specific weight (computed from balanced dataset)
- `γ`: Focusing parameter (3.0 for aggressive focusing)


## Visualization

Generate analysis plots:

```bash
python src/analyze_results.py \
    --predictions predictions.csv \
    --ground_truth Task_B/validation.parquet \
    --output_dir analysis/
```

Generates:
- Confusion matrix
- Per-class F1 bar chart
- Training history curves
- Error analysis by language/generator


## Acknowledgments

- **CodeBERT**: Microsoft Research for the pre-trained CodeBERT model
- **SemEval-2026**: Task organizers for the dataset and competition
- **Focal Loss**: Lin et al. (2017) for the focal loss formulation
- **MBZUAI**: Mohamed bin Zayed University of Artificial Intelligence for computational resources

## References

1. Feng et al. (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages. EMNLP.
2. Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
3. [SemEval-2026 Task 13](https://www.kaggle.com/competitions/sem-eval-2026-task-13-subtask-b/leaderboard): Multi-Class Authorship Detection of AI-Generated Code.
