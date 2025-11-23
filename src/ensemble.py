import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import f1_score
import json

def load_predictions(file_path):
    """Load predictions from CSV file"""
    df = pd.read_csv(file_path)
    return df['label'].values

def simple_majority_voting(predictions_list):
    """Simple majority voting ensemble"""
    predictions_array = np.array(predictions_list)
    # Get mode (most common prediction) for each sample
    ensemble = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 
        axis=0, 
        arr=predictions_array
    )
    return ensemble

def weighted_voting(predictions_list, weights):
    """Weighted voting ensemble"""
    if len(predictions_list) != len(weights):
        raise ValueError("Number of predictions must match number of weights")
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    num_classes = 11
    num_samples = len(predictions_list[0])
    
    # Create vote matrix
    vote_matrix = np.zeros((num_samples, num_classes))
    
    for preds, weight in zip(predictions_list, weights):
        for i, pred in enumerate(preds):
            vote_matrix[i, pred] += weight
    
    # Get class with highest weighted vote
    ensemble = np.argmax(vote_matrix, axis=1)
    return ensemble

def soft_voting(predictions_proba_list, weights=None):
    """Soft voting using prediction probabilities (if available)"""
    if weights is None:
        weights = np.ones(len(predictions_proba_list)) / len(predictions_proba_list)
    else:
        weights = np.array(weights) / np.sum(weights)
    
    # Average probabilities
    avg_proba = np.average(predictions_proba_list, axis=0, weights=weights)
    ensemble = np.argmax(avg_proba, axis=1)
    return ensemble

def rank_based_voting(predictions_list, validation_f1_scores):
    """Voting weighted by model performance"""
    # Sort models by F1 score
    sorted_indices = np.argsort(validation_f1_scores)[::-1]
    
    # Assign exponentially decaying weights
    weights = np.exp(-0.5 * np.arange(len(predictions_list)))
    weights = weights / np.sum(weights)
    
    # Reorder predictions and weights
    sorted_predictions = [predictions_list[i] for i in sorted_indices]
    
    return weighted_voting(sorted_predictions, weights)

def evaluate_ensemble(predictions, ground_truth):
    """Evaluate ensemble predictions"""
    macro_f1 = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    per_class_f1 = f1_score(ground_truth, predictions, average=None, zero_division=0, labels=range(11))
    accuracy = (predictions == ground_truth).mean()
    
    return {
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1.tolist(),
        'accuracy': accuracy
    }

def main():
    parser = argparse.ArgumentParser(description='Ensemble multiple model predictions')
    parser.add_argument('--predictions', nargs='+', required=True,
                        help='List of prediction CSV files')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Weights for weighted voting (must match number of predictions)')
    parser.add_argument('--method', type=str, default='majority', 
                        choices=['majority', 'weighted', 'rank'],
                        help='Ensemble method')
    parser.add_argument('--validation_scores', nargs='+', type=float, default=None,
                        help='Validation F1 scores for rank-based voting')
    parser.add_argument('--ground_truth', type=str, default=None,
                        help='Ground truth file for evaluation (optional)')
    parser.add_argument('--output', type=str, default='submission_ensemble.csv',
                        help='Output submission file')
    parser.add_argument('--analysis_output', type=str, default='ensemble_analysis.json',
                        help='Output file for analysis')
    args = parser.parse_args()
    
    print(f"Loading {len(args.predictions)} prediction files...")
    
    # Load all predictions
    predictions_list = []
    ids = None
    
    for pred_file in args.predictions:
        print(f"  - {pred_file}")
        df = pd.read_csv(pred_file)
        predictions_list.append(df['label'].values)
        
        if ids is None:
            ids = df['id'].values
    
    # Verify all predictions have same length
    lengths = [len(p) for p in predictions_list]
    if len(set(lengths)) > 1:
        raise ValueError(f"Predictions have different lengths: {lengths}")
    
    print(f"\nLoaded {len(predictions_list)} models with {len(predictions_list[0])} samples each")
    
    # Create ensemble based on method
    print(f"\nCreating ensemble using '{args.method}' method...")
    
    if args.method == 'majority':
        ensemble_predictions = simple_majority_voting(predictions_list)
        print("Using simple majority voting")
        
    elif args.method == 'weighted':
        if args.weights is None:
            # Equal weights
            weights = [1.0] * len(predictions_list)
            print("No weights provided, using equal weights")
        else:
            weights = args.weights
            if len(weights) != len(predictions_list):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of predictions ({len(predictions_list)})")
        
        ensemble_predictions = weighted_voting(predictions_list, weights)
        print(f"Using weighted voting with weights: {weights}")
        
    elif args.method == 'rank':
        if args.validation_scores is None:
            raise ValueError("Rank-based voting requires --validation_scores")
        if len(args.validation_scores) != len(predictions_list):
            raise ValueError(f"Number of validation scores must match number of predictions")
        
        ensemble_predictions = rank_based_voting(predictions_list, args.validation_scores)
        print(f"Using rank-based voting with F1 scores: {args.validation_scores}")
    
    # Create submission file
    submission_df = pd.DataFrame({
        'id': ids,
        'label': ensemble_predictions
    })
    
    submission_df.to_csv(args.output, index=False)
    print(f"\nEnsemble predictions saved to: {args.output}")
    
    # Print prediction distribution
    print("\nEnsemble prediction distribution:")
    print(submission_df['label'].value_counts().sort_index())
    
    # Compare with individual models
    print("\nPrediction agreement analysis:")
    agreements = []
    for i, preds in enumerate(predictions_list):
        agreement = (preds == ensemble_predictions).mean()
        agreements.append(agreement)
        print(f"  Model {i+1} agreement with ensemble: {agreement:.3f}")
    
    # Evaluate if ground truth is provided
    analysis = {
        'method': args.method,
        'num_models': len(predictions_list),
        'prediction_distribution': submission_df['label'].value_counts().sort_index().to_dict(),
        'model_agreements': agreements
    }
    
    if args.ground_truth:
        print(f"\nEvaluating against ground truth: {args.ground_truth}")
        
        # Load ground truth
        if args.ground_truth.endswith('.parquet'):
            gt_df = pd.read_parquet(args.ground_truth)
        else:
            gt_df = pd.read_csv(args.ground_truth)
        
        ground_truth = gt_df['label'].values
        
        # Evaluate ensemble
        ensemble_metrics = evaluate_ensemble(ensemble_predictions, ground_truth)
        
        print(f"\nEnsemble Performance:")
        print(f"  Macro F1: {ensemble_metrics['macro_f1']:.4f}")
        print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
        
        # Evaluate individual models
        print(f"\nIndividual Model Performance:")
        individual_metrics = []
        for i, preds in enumerate(predictions_list):
            metrics = evaluate_ensemble(preds, ground_truth)
            individual_metrics.append(metrics)
            print(f"  Model {i+1} - Macro F1: {metrics['macro_f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        # Calculate improvement
        best_individual_f1 = max(m['macro_f1'] for m in individual_metrics)
        improvement = ensemble_metrics['macro_f1'] - best_individual_f1
        
        print(f"\nEnsemble Improvement: {improvement:+.4f} over best individual model")
        
        # Add to analysis
        analysis['ensemble_metrics'] = {
            'macro_f1': float(ensemble_metrics['macro_f1']),
            'accuracy': float(ensemble_metrics['accuracy']),
            'per_class_f1': ensemble_metrics['per_class_f1']
        }
        analysis['individual_metrics'] = [
            {
                'macro_f1': float(m['macro_f1']),
                'accuracy': float(m['accuracy'])
            }
            for m in individual_metrics
        ]
        analysis['improvement'] = float(improvement)
    
    # Save analysis
    with open(args.analysis_output, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis saved to: {args.analysis_output}")
    print("\nDone!")

if __name__ == '__main__':
    main()