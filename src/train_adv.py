import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from collections import Counter
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)

scaler = GradScaler()

class CodeDataset(Dataset):
    """Enhanced dataset with metadata features"""
    def __init__(self, df, tokenizer, max_length=512, use_metadata=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_metadata = use_metadata
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = str(row['code'])
        label = int(row['label'])
        
        # Add metadata as prefix if available
        if self.use_metadata and 'language' in self.df.columns:
            language = str(row.get('language', ''))
            code_with_meta = f"# Language: {language}\n{code}"
        else:
            code_with_meta = code
        
        encoding = self.tokenizer(
            code_with_meta,
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

class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_weighted_sampler(df, num_classes=11):
    """Create weighted sampler for balanced training"""
    labels = df['label'].values
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def train_epoch_focal(model, dataloader, optimizer, scheduler, device, criterion, accumulation_steps=4):
    """Train with focal loss and gradient accumulation"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    optimizer.zero_grad()    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        loss = criterion(outputs.logits, labels)
        # normalize the loss
        loss = loss / accumulation_steps
        
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    avg_loss = total_loss / len(dataloader)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    
    return avg_loss, f1_macro

def evaluate(model, dataloader, device, num_classes=11):
    """Evaluate the model"""
    model.eval()
    predictions = []
    true_labels = []
    
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
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0, labels=range(num_classes))
    
    return f1_macro, f1_per_class, predictions, true_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base')
    parser.add_argument('--train_file', type=str, default=r'C:\Users\PC\OneDrive\Documents\uni\NLP_Assignment2\Task_B\train_balanced.parquet')
    parser.add_argument('--val_file', type=str, default=r'C:\Users\PC\OneDrive\Documents\uni\NLP_Assignment2\Task_B\validation.parquet')    
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--output_dir', type=str, default='outputs_advanced')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss')
    parser.add_argument('--use_weighted_sampler', action='store_true', help='Use weighted sampler')
    parser.add_argument('--use_metadata', action='store_true', help='Include metadata in input')
    parser.add_argument('--focal_gamma', type=float, default=3.0)
    parser.add_argument('--sample_train', type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_df = pd.read_parquet(args.train_file)
    val_df = pd.read_parquet(args.val_file)
    
    if args.sample_train:
        train_df = train_df.sample(n=min(args.sample_train, len(train_df)), random_state=42)
    
    print(f'Train size: {len(train_df)}, Val size: {len(val_df)}')
    print(f'Label distribution:\n{train_df["label"].value_counts().sort_index()}')
    
    # Load model
    print(f'Loading model: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=args.num_classes
    )
    model.to(device)
    
    # Create datasets
    train_dataset = CodeDataset(train_df, tokenizer, args.max_length, args.use_metadata)
    val_dataset = CodeDataset(val_df, tokenizer, args.max_length, args.use_metadata)
    
    # Create dataloaders with optional weighted sampling
    if args.use_weighted_sampler:
        sampler = get_weighted_sampler(train_df, args.num_classes)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Setup loss function
    if args.use_focal_loss:
        # Compute class weights
        labels = train_df['label'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f'Using Focal Loss with gamma={args.focal_gamma}')
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps), # try 20%
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    history = []
    
    for epoch in range(args.epochs):
        print(f'\n=== Epoch {epoch + 1}/{args.epochs} ===')
        
        train_loss, train_f1 = train_epoch_focal(model, train_loader, optimizer, scheduler, device, criterion)
        val_f1, val_f1_per_class, _, _ = evaluate(model, val_loader, device, args.num_classes)
        
        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val F1: {val_f1:.4f}')
        print(f'Val F1 per class: {val_f1_per_class}')
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_f1': train_f1,
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