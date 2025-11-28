import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import f1_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Label mappings
ID_TO_LABEL = {
    "0": "human",
    "1": "deepseek",
    "2": "qwen",
    "3": "01-ai",
    "4": "bigcode",
    "5": "gemma",
    "6": "phi",
    "7": "meta-llama",
    "8": "ibm-granite",
    "9": "mistral",
    "10": "openai"
}

LABEL_TO_ID = {v: int(k) for k, v in ID_TO_LABEL.items()}

class CodeDataset(Dataset):
    """Enhanced dataset for code classification with metadata"""
    def __init__(self, df, tokenizer, max_length=512, include_metadata=True, label_column='label'):
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
        
        # Optionally prepend language/domain info
        if self.include_metadata and 'language' in self.df.columns:
            language = str(row.get('language', 'unknown'))
            code = f"# Language: {language}\n{code}"
        
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
            'id': idx if 'id' not in self.df.columns else row['id']
        }
        
        # Include labels if available (training/validation)
        if self.label_column in self.df.columns:
            label = row[self.label_column]
            
            # Convert string label to id if necessary
            if isinstance(label, str):
                label_id = LABEL_TO_ID.get(label, 0)
            else:
                label_id = int(label)
            
            result['labels'] = torch.tensor(label_id, dtype=torch.long)
        
        return result

def train_epoch(model, dataloader, optimizer, scheduler, device):
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
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
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
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, 
                            labels=range(num_classes))
    
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
            all_ids.extend(batch['id'].cpu().numpy() if isinstance(batch['id'], torch.Tensor) 
                          else batch['id'])
    
    return all_ids, all_predictions

def print_class_distribution(df, label_column='label', mode='train'):
    """Print distribution of labels"""
    print(f"\n{mode.upper()} SET LABEL DISTRIBUTION:")
    
    # Count by numeric label
    counts = df[label_column].value_counts().sort_index()
    total = len(df)
    
    for label_id in sorted(counts.index):
        count = counts[label_id]
        percentage = 100 * count / total
        label_name = ID_TO_LABEL.get(str(int(label_id)), "unknown")
        print(f"  {label_id:2d} ({label_name:15s}): {count:5d} ({percentage:5.2f}%)")
    
    print(f"  Total: {total}")

def main():
    parser = argparse.ArgumentParser(
        description='SemEval-2026 Task 13 Subtask B - Competition Rules Compliant'
    )
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'], 
                        default='train', help='Mode: train, evaluate, or predict')
    
    # Model arguments (ONLY general-purpose models allowed)
    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base',
                        help='General-purpose pretrained model (CodeBERT, Unixcoder, GraphCodeBERT, StarCoder, etc.)')
    parser.add_argument('--output_dir', type=str, default='model_checkpoint',
                        help='Directory to save model')
    
    # Data arguments (ONLY official SemEval data allowed)
    parser.add_argument('--train_file', type=str, help='Official SemEval training file path')
    parser.add_argument('--val_file', type=str, help='Official SemEval validation file path')
    parser.add_argument('--test_file', type=str, help='Official SemEval test file path')
    parser.add_argument('--model_path', type=str, help='Path to trained model for prediction')
    
    # Training arguments
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default='submission.csv',
                        help='Output submission file')
    parser.add_argument('--include_metadata', action='store_true', 
                        help='Include language metadata in input')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Random seed: {args.seed}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print(f'\nLoading tokenizer and model: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_classes
    )
    model.to(device)
    
    if args.mode == 'train':
        print('\n=== TRAINING MODE ===')
        print('COMPLIANCE CHECK: Using ONLY official SemEval training data')
        
        # Load ONLY official data
        print('\nLoading official SemEval training and validation data...')
        if not args.train_file or not args.val_file:
            raise ValueError('--train_file and --val_file required for training mode')
        
        train_df = pd.read_parquet(args.train_file)
        val_df = pd.read_parquet(args.val_file)
        
        print(f'\nTrain size: {len(train_df)}, Val size: {len(val_df)}')
        
        # Print label distributions
        print_class_distribution(train_df, mode='train')
        print_class_distribution(val_df, mode='validation')
        
        # Create datasets and dataloaders
        train_dataset = CodeDataset(train_df, tokenizer, args.max_length, args.include_metadata)
        val_dataset = CodeDataset(val_df, tokenizer, args.max_length, args.include_metadata)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        # Training loop
        best_f1 = 0
        history = []
        
        print(f'\n--- Training Configuration ---')
        print(f'Learning rate: {args.lr}')
        print(f'Batch size: {args.batch_size}')
        print(f'Epochs: {args.epochs}')
        print(f'Max length: {args.max_length}')
        print(f'Model: {args.model_name}')
        print(f'---\n')
        
        for epoch in range(args.epochs):
            print(f'\n=== Epoch {epoch + 1}/{args.epochs} ===')
            
            train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_f1, val_f1_per_class, _, _ = evaluate(model, val_loader, device, args.num_classes)
            
            print(f'Train Loss: {train_loss:.4f}, Train F1 (macro): {train_f1:.4f}')
            print(f'Val F1 (macro): {val_f1:.4f}')
            print(f'Val F1 per class:')
            for label_id, f1 in enumerate(val_f1_per_class):
                label_name = ID_TO_LABEL.get(str(label_id), "unknown")
                print(f'  {label_id:2d} ({label_name:15s}): {f1:.4f}')
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_f1': float(train_f1),
                'val_f1': float(val_f1),
                'val_f1_per_class': [float(f) for f in val_f1_per_class]
            })
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                model_path = os.path.join(args.output_dir, 'best_model')
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                print(f'Saved best model with F1: {best_f1:.4f}')
        
        # Save training history
        with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f'\nTraining completed! Best Val F1 (macro): {best_f1:.4f}')
        print(f'Model saved to: {os.path.join(args.output_dir, "best_model")}')
        
    elif args.mode == 'evaluate':
        print('\n=== EVALUATION MODE ===')
        
        # Load validation data
        if not args.val_file:
            raise ValueError('--val_file required for evaluation mode')
        
        val_df = pd.read_parquet(args.val_file)
        print(f'Validation size: {len(val_df)}')
        
        # Load trained model
        print(f'Loading model from: {args.model_path}')
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model.to(device)
        
        # Create dataset and dataloader
        val_dataset = CodeDataset(val_df, tokenizer, args.max_length, args.include_metadata)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Evaluate
        val_f1, val_f1_per_class, preds, labels = evaluate(model, val_loader, device, args.num_classes)
        
        print(f'\n--- Validation Results ---')
        print(f'Macro F1: {val_f1:.4f}')
        print(f'\nPer-class F1:')
        for label_id, f1 in enumerate(val_f1_per_class):
            label_name = ID_TO_LABEL.get(str(label_id), "unknown")
            print(f'  {label_id:2d} ({label_name:15s}): {f1:.4f}')
        
        print(f'\nClassification Report:')
        target_names = [ID_TO_LABEL.get(str(i), "unknown") for i in range(args.num_classes)]
        print(classification_report(labels, preds, target_names=target_names, zero_division=0))
        
    elif args.mode == 'predict':
        print('\n=== PREDICTION MODE ===')
        
        # Load test data
        if not args.test_file:
            raise ValueError('--test_file required for prediction mode')
        
        print('Loading test data...')
        test_df = pd.read_parquet(args.test_file)
        print(f'Test size: {len(test_df)}')
        
        # Load trained model
        print(f'Loading model from: {args.model_path}')
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model.to(device)
        
        # Create dataset and dataloader
        test_dataset = CodeDataset(test_df, tokenizer, args.max_length, args.include_metadata)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Generate predictions
        print('Generating predictions...')
        ids, predictions = predict(model, test_loader, device)
        
        # Create submission file
        submission_df = pd.DataFrame({
            'id': ids,
            'label': predictions
        })
        
        # Sort by id
        submission_df = submission_df.sort_values('id').reset_index(drop=True)
        
        # Save submission
        submission_df.to_csv(args.output_file, index=False)
        print(f'\nPredictions saved to: {args.output_file}')
        
        print(f'\nPrediction distribution:')
        dist = submission_df['label'].value_counts().sort_index()
        for label_id, count in dist.items():
            label_name = ID_TO_LABEL.get(str(label_id), "unknown")
            percentage = 100 * count / len(submission_df)
            print(f'  {label_id:2d} ({label_name:15s}): {count:5d} ({percentage:5.2f}%)')

if __name__ == '__main__':
    main()