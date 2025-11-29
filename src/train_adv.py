import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

# Label mappings
ID_TO_LABEL = {
    0: "human", 1: "deepseek", 2: "qwen", 3: "01-ai", 4: "bigcode",
    5: "gemma", 6: "phi", 7: "meta-llama", 8: "ibm-granite",
    9: "mistral", 10: "openai"
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


class FocalLoss(torch.nn.Module):
    """Focal Loss for handling extreme class imbalance"""
    def __init__(self, alpha, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=None)
        
        # Get probabilities
        p = torch.softmax(inputs, dim=1)
        pt = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weight for class balance
        alpha_t = self.alpha[targets]
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class CodeDataset(Dataset):
    """Dataset for code classification"""
    def __init__(self, df, tokenizer, max_length=512, include_metadata=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_metadata = include_metadata
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = str(row['code'])
        
        if self.include_metadata and 'language' in self.df.columns:
            language = str(row.get('language', 'unknown'))
            code = f"Language: {language}. {code}"

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
        
        if 'label' in self.df.columns:
            label = row['label']
            label_id = int(label)
            result['labels'] = torch.tensor(label_id, dtype=torch.long)
        
        return result


def get_balanced_data(train_df, strategy='undersample'):
    """Get balanced data using different strategies"""
    
    if strategy == 'undersample':
        # Undersample majority class
        majority_df = train_df[train_df['label'] == 0]
        minority_df = train_df[train_df['label'] != 0]
        target_count = len(minority_df)
        undersampled_majority = majority_df.sample(n=target_count, random_state=42)
        balanced_df = pd.concat([undersampled_majority, minority_df], ignore_index=True)
        
    elif strategy == 'no_balance':
        # Use all data with weights
        balanced_df = train_df
    
    # Calculate weights
    label_counts = balanced_df['label'].value_counts()
    num_samples = label_counts.sum()
    class_weights = num_samples / (len(label_counts) * label_counts.sort_index())
    sample_weights = balanced_df['label'].apply(lambda x: class_weights[x]).values
    
    print(f"\nData Strategy: {strategy}")
    print(f"Training Samples: {len(balanced_df)}")
    for label_id in sorted(label_counts.index):
        count = label_counts[label_id]
        percentage = 100 * count / len(balanced_df)
        label_name = ID_TO_LABEL.get(label_id, "unknown")
        print(f"  {label_id:2d} ({label_name:15s}): {count:6d} ({percentage:5.2f}%)")
    
    return balanced_df, sample_weights, class_weights


def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """Train for one epoch"""
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
        
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
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
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=range(num_classes))
    
    return macro_f1, per_class_f1, all_preds, all_labels


def predict(model, dataloader, device):
    """Generate predictions"""
    model.eval()
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_ids.extend(batch['id'].cpu().numpy() if isinstance(batch['id'], torch.Tensor) else batch['id'])
    
    return all_ids, all_predictions


def main():
    parser = argparse.ArgumentParser(
        description='SemEval-2026 Task 13 Subtask B: Improved Training with Multiple Models'
    )
    
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'], default='train')
    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base',
                        help='Model: codebert-base, unixcoder-base, graphcodebert-base')
    parser.add_argument('--output_dir', type=str, default='model_checkpoint')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    
    parser.add_argument('--train_file', type=str, help='Training file')
    parser.add_argument('--val_file', type=str, help='Validation file')
    parser.add_argument('--test_file', type=str, help='Test file')
    
    # Training hyperparameters (IMPROVED DEFAULTS)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)  # Reduced from 32
    parser.add_argument('--epochs', type=int, default=10)     # Increased from 5
    parser.add_argument('--lr', type=float, default=1e-5)     # Reduced from 2e-5
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    
    # Focal loss parameters
    parser.add_argument('--focal_gamma', type=float, default=2.0)  # Reduced from 3.0
    parser.add_argument('--focal_strategy', type=str, choices=['undersample', 'no_balance'], 
                        default='undersample')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include_metadata', action='store_true', default=True)
    parser.add_argument('--output_file', type=str, default='submission.csv')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Model: {args.model_name}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.mode == 'train':
        print('\n=== TRAINING MODE (Improved) ===')
        
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=11)
        model.to(device)
        
        train_df = pd.read_parquet(args.train_file)
        val_df = pd.read_parquet(args.val_file)
        
        # Get balanced data
        balanced_train_df, sample_weights, class_weights = get_balanced_data(train_df, args.focal_strategy)
        
        # Create datasets
        train_dataset = CodeDataset(balanced_train_df, tokenizer, args.max_length, args.include_metadata)
        val_dataset = CodeDataset(val_df, tokenizer, args.max_length, args.include_metadata)
        
        # Sampler
        train_sampler = WeightedRandomSampler(sample_weights, len(balanced_train_df), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2)
        
        # Loss and optimizer
        class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=args.focal_gamma)
        
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps
        )
        
        print(f'\n--- Training Config ---')
        print(f'Batch Size: {args.batch_size}')
        print(f'Epochs: {args.epochs}')
        print(f'LR: {args.lr}')
        print(f'Focal Gamma: {args.focal_gamma}')
        print(f'Max Length: {args.max_length}')
        print(f'Total Steps: {total_steps}')
        print(f'---\n')
        
        best_f1 = 0
        patience = 3
        patience_counter = 0
        
        for epoch in range(args.epochs):
            print(f'\n=== Epoch {epoch + 1}/{args.epochs} ===')
            
            train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device, criterion)
            val_f1, val_f1_per_class, _, _ = evaluate(model, val_loader, device, 11)
            
            print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
            print(f'Val F1: {val_f1:.4f}')
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                model_path = os.path.join(args.output_dir, 'best_model')
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                print(f'✓ Best model saved: F1={best_f1:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
        
        print(f'\n✓ Training complete! Best F1: {best_f1:.4f}')
        
    elif args.mode == 'evaluate':
        print('\n=== EVALUATION MODE ===')
        
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        model.to(device)
        
        val_df = pd.read_parquet(args.val_file)
        val_dataset = CodeDataset(val_df, tokenizer, args.max_length, args.include_metadata)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2)
        
        val_f1, val_f1_per_class, preds, labels = evaluate(model, val_loader, device, 11)
        
        print(f'\nMacro F1: {val_f1:.4f}')
        print(f'\nPer-class F1:')
        for i, f1 in enumerate(val_f1_per_class):
            print(f'  {i:2d} ({ID_TO_LABEL[i]:15s}): {f1:.4f}')
        
    elif args.mode == 'predict':
        print('\n=== PREDICTION MODE ===')
        
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        model.to(device)
        
        test_df = pd.read_parquet(args.test_file)
        test_dataset = CodeDataset(test_df, tokenizer, args.max_length, args.include_metadata)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2)
        
        ids, predictions = predict(model, test_loader, device)
        
        submission_df = pd.DataFrame({'id': ids, 'label': predictions})
        submission_df = submission_df.sort_values('id').reset_index(drop=True)
        submission_df.to_csv(args.output_file, index=False)
        
        print(f'\nSaved: {args.output_file}')
        print(f'Predictions:\n{submission_df["label"].value_counts().sort_index()}')


if __name__ == '__main__':
    main()