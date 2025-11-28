import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import log
from collections import Counter

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

from sklearn.metrics import f1_score, classification_report

import warnings
warnings.filterwarnings('ignore')

# --- 1. MAPPINGS & CUSTOM LOSS FUNCTION ---

# Label mappings
ID_TO_LABEL = {
    0: "human", 1: "deepseek", 2: "qwen", 3: "01-ai", 4: "bigcode",
    5: "gemma", 6: "phi", 7: "meta-llama", 8: "ibm-granite",
    9: "mistral", 10: "openai"
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation used for extreme class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # alpha is the class weight vector
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits (B, C), targets are class indices (B,)
        
        # 1. Standard Cross-Entropy:
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. Compute probabilities (p_t) and pt = p_t
        pt = torch.exp(-CE_loss) 
        
        # 3. Compute the modulating factor (1 - p_t)^gamma
        F_mod = (1 - pt) ** self.gamma
        
        # 4. Compute alpha weights (alpha_t)
        # alpha is a vector (C,)
        alpha_t = self.alpha[targets] 
        
        # 5. Full Focal Loss: alpha_t * (1 - p_t)^gamma * CE_loss
        loss = alpha_t * F_mod * CE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


# --- 2. DATA UTILITIES ---

class CodeDataset(Dataset):
    """Dataset for code classification (unchanged)"""
    def __init__(self, df, tokenizer, max_length=512, include_metadata=False, label_column='label'):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_metadata = include_metadata
        self.label_column = label_column
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = str(row['code'])
        
        # Adding metadata as a stronger feature (Richness of Features)
        if self.include_metadata and 'language' in self.df.columns:
            language = str(row.get('language', 'unknown'))
            # Format: [CLS] The following code is in python. [SEP] CODE ...
            code = f"The following code is in {language}. {self.tokenizer.sep_token} {code}"

        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'id': idx
        }
        
        if self.label_column in self.df.columns:
            label = row[self.label_column]
            label_id = int(label)
            result['labels'] = torch.tensor(label_id, dtype=torch.long)
        
        return result

def get_balanced_train_data(train_df):
    """
    Implements targeted undersampling for the majority class (Human, ID 0) 
    to create a balanced training subset (~116k samples).
    """
    
    # 1. Separate Majority and Minority
    majority_df = train_df[train_df['label'] == 0]
    minority_df = train_df[train_df['label'] != 0]
    
    # Target size for the majority class is the total size of all minority classes
    target_count = len(minority_df) # 57,904 samples
    
    # 2. Undersample Majority
    # The '107k' in the report is an approximation. Using a 1:1 ratio is cleaner.
    if target_count < len(majority_df):
        undersampled_majority = majority_df.sample(n=target_count, random_state=42)
    else:
        undersampled_majority = majority_df
        
    # 3. Combine to form the Balanced Subset
    balanced_df = pd.concat([undersampled_majority, minority_df], ignore_index=True)
    
    # 4. Calculate Weights for Weighted Random Sampler
    # Weights are inversely proportional to class frequency in the balanced subset
    label_counts = balanced_df['label'].value_counts()
    num_samples = label_counts.sum()
    class_weights = num_samples / label_counts.sort_index()
    sample_weights = balanced_df['label'].apply(lambda x: class_weights[x]).values
    
    print(f"\nCreated Balanced Training Subset (N={len(balanced_df)}):")
    for label_id in sorted(label_counts.index):
        count = label_counts[label_id]
        percentage = 100 * count / len(balanced_df)
        label_name = ID_TO_LABEL.get(label_id, "unknown")
        print(f"  {label_id:2d} ({label_name:15s}): {count:5d} ({percentage:5.2f}%)")
        
    return balanced_df, sample_weights, class_weights


# --- 3. TRAINING & EVALUATION LOOPS ---

def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """Train for one epoch using Focal Loss"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use Custom Focal Loss
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
    
    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, macro_f1

def evaluate(model, dataloader, device, num_classes=11):
    """Evaluate on validation set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, 
                             labels=range(num_classes))
    
    return macro_f1, per_class_f1, all_preds, all_labels

# --- 4. MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(
        description='SemEval-2026 Task 13 Subtask B: Balanced Training with Focal Loss'
    )
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'], 
                         default='train', help='Mode: train, evaluate, or predict')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base',
                         help='General-purpose pretrained model')
    parser.add_argument('--output_dir', type=str, default='model_checkpoint',
                         help='Directory to save model')
    parser.add_argument('--model_path', type=str, help='Path to trained model for evaluation/prediction')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, help='Official SemEval training file path')
    parser.add_argument('--val_file', type=str, help='Official SemEval validation file path')
    parser.add_argument('--test_file', type=str, help='Official SemEval test file path')
    
    # Training arguments
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--focal_gamma', type=float, default=3.0, help='Gamma parameter for Focal Loss')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--include_metadata', action='store_true', 
                         help='Include language metadata in input (Recommended)')

    # Output arguments
    parser.add_argument('--output_file', type=str, default='submission.csv',
                         help='Output submission file')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print(f'\nLoading tokenizer and model: {args.model_name}')
    if args.mode == 'train':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=args.num_classes
        )
    else: # Load trained model for evaluate/predict
        if not args.model_path:
             raise ValueError('--model_path required for evaluate/predict mode')
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        
    model.to(device)

    # --- TRAIN MODE ---
    if args.mode == 'train':
        print('\n=== TRAINING MODE: Balanced Training with Focal Loss ===')
        if not args.train_file or not args.val_file:
            raise ValueError('--train_file and --val_file required for training mode')
        
        train_df = pd.read_parquet(args.train_file)
        val_df = pd.read_parquet(args.val_file)
        
        # 1. Undersample Majority Class & Get Weights
        balanced_train_df, sample_weights, class_weights = get_balanced_train_data(train_df)
        
        # 2. Setup Sampler and Loaders
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(balanced_train_df),
            replacement=True
        )
        
        train_dataset = CodeDataset(balanced_train_df, tokenizer, args.max_length, args.include_metadata)
        val_dataset = CodeDataset(val_df, tokenizer, args.max_length, args.include_metadata)
        
        # Use the sampler for the training dataloader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # 3. Setup Loss, Optimizer, and Scheduler
        class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=args.focal_gamma)
        
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )
        
        # 4. Training Loop
        best_f1 = 0
        
        print(f'\nFocal Loss Gamma: {args.focal_gamma}')
        print(f'Batch Size: {args.batch_size}')
        print(f'Epochs: {args.epochs}')
        print(f'Total Training Steps: {total_steps}')
        
        for epoch in range(args.epochs):
            print(f'\n=== Epoch {epoch + 1}/{args.epochs} ===')
            
            train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device, criterion)
            val_f1, val_f1_per_class, _, _ = evaluate(model, val_loader, device, args.num_classes)
            
            print(f'Train Loss: {train_loss:.4f}, Train F1 (macro): {train_f1:.4f}')
            print(f'Val F1 (macro): {val_f1:.4f}')
            print(f'Val F1 per class:')
            for label_id, f1 in ID_TO_LABEL.items():
                print(f'  {label_id:2d} ({ID_TO_LABEL[label_id]:15s}): {val_f1_per_class[label_id]:.4f}')
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                model_path = os.path.join(args.output_dir, 'best_model')
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                print(f'Saved best model with F1: {best_f1:.4f}')
        
        print(f'\nTraining completed! Best Val F1 (macro): {best_f1:.4f}')

    # --- EVALUATE MODE ---
    elif args.mode == 'evaluate':
        print('\n=== EVALUATION MODE ===')
        if not args.val_file:
             raise ValueError('--val_file required for evaluation mode')
        
        val_df = pd.read_parquet(args.val_file)
        val_dataset = CodeDataset(val_df, tokenizer, args.max_length, args.include_metadata)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        val_f1, val_f1_per_class, preds, labels = evaluate(model, val_loader, device, args.num_classes)
        
        print(f'\n--- Validation Results ---')
        print(f'Macro F1: {val_f1:.4f}')
        target_names = [ID_TO_LABEL.get(i, "unknown") for i in range(args.num_classes)]
        print(f'\nClassification Report:')
        print(classification_report(labels, preds, target_names=target_names, zero_division=0))

    # --- PREDICT MODE ---
    elif args.mode == 'predict':
        print('\n=== PREDICTION MODE ===')
        if not args.test_file:
            raise ValueError('--test_file required for prediction mode')
        
        test_df = pd.read_parquet(args.test_file)
        test_dataset = CodeDataset(test_df, tokenizer, args.max_length, args.include_metadata)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Generate predictions
        ids, predictions = predict(model, test_loader, device)
        
        # Create submission file
        submission_df = pd.DataFrame({
            'id': ids,
            'label': predictions
        })
        submission_df = submission_df.sort_values('id').reset_index(drop=True)
        submission_df.to_csv(args.output_file, index=False)
        
        print(f'\nPredictions saved to: {args.output_file}')
        print(f'Prediction distribution:\n{submission_df["label"].value_counts().sort_index()}')

if __name__ == '__main__':
    main()