import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score


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
            'id': idx
        }

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
            all_ids.extend(batch['id'].numpy())
    
    return all_ids, all_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test file to predict')
    parser.add_argument('--output_file', type=str, default='submission.csv',
                        help='Output submission file')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load test data
    print('Loading test data...')
    test_df = pd.read_parquet(args.test_file)
    print(f'Test size: {len(test_df)}')
    
    # Load tokenizer and model
    print(f'Loading model from: {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    
    # Create dataset and dataloader
    test_dataset = CodeDataset(test_df, tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Generate predictions
    print('Generating predictions...')
    ids, predictions = predict(model, test_loader, device)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'id': ids,
        'label': predictions
    })
    
    # Sort by id to ensure correct order
    submission_df = submission_df.sort_values('id').reset_index(drop=True)
    
    # Save submission
    submission_df.to_csv(args.output_file, index=False)
    print(f'Predictions saved to: {args.output_file}')
    print(f'Prediction distribution:\n{submission_df["label"].value_counts().sort_index()}')

if __name__ == '__main__':
    main()