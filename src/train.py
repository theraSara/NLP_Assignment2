import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import argparse
import json
from datetime import datetime

class CodeDataset(Dataset):
    """Dataset for code classification"""
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        code = str(self.df.iloc[idx]['code'])
        label = int(self.df.iloc[idx]['label'])
        
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def calculate_metrics(predictions, labels, num_classes=11):
    """Calculate macro F1 score"""
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0, labels=range(num_classes))
    return f1_macro, f1_per_class

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
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
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    f1_macro, _ = calculate_metrics(predictions, true_labels)
    
    return avg_loss, f1_macro

def evaluate(model, dataloader, device, num_classes=11):
    """Evaluate the model"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1_macro, f1_per_class = calculate_metrics(predictions, true_labels, num_classes)
    
    return avg_loss, f1_macro, f1_per_class, predictions, true_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base', 
                        help='Pretrained model name')
    parser.add_argument('--train_file', type=str, default=r'C:\Users\PC\OneDrive\Documents\uni\NLP_Assignment2\Task_B\train.parquet')
    parser.add_argument('--val_file', type=str, default=r'C:\Users\PC\OneDrive\Documents\uni\NLP_Assignment2\Task_B\validation.parquet')
    parser.add_argument('--test_file', type=str, default=r'C:\Users\PC\OneDrive\Documents\uni\NLP_Assignment2\Task_B\test_sample.parquet')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--sample_train', type=int, default=None, 
                        help='Sample size for quick testing')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_df = pd.read_parquet(args.train_file)
    val_df = pd.read_parquet(args.val_file)
    
    if args.sample_train:
        train_df = train_df.sample(n=min(args.sample_train, len(train_df)), random_state=42)
        print(f'Using {len(train_df)} samples for training')
    
    print(f'Train size: {len(train_df)}, Val size: {len(val_df)}')
    print(f'Label distribution in train:\n{train_df["label"].value_counts().sort_index()}')
    
    # Load tokenizer and model
    print(f'Loading model: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=args.num_classes,
        trust_remote_code=False
    )
    model.to(device)
    
    # Create datasets
    train_dataset = CodeDataset(train_df, tokenizer, args.max_length)
    val_dataset = CodeDataset(val_df, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    history = []
    
    for epoch in range(args.epochs):
        print(f'\n=== Epoch {epoch + 1}/{args.epochs} ===')
        
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_f1, val_f1_per_class, _, _ = evaluate(model, val_loader, device, args.num_classes)
        
        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        print(f'Val F1 per class: {val_f1_per_class}')
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_f1': val_f1
        })
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            model_path = os.path.join(args.output_dir, 'best_model')
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f'Saved best model with F1: {best_f1:.4f}')
    
    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\nTraining completed! Best Val F1: {best_f1:.4f}')

if __name__ == '__main__':
    main()