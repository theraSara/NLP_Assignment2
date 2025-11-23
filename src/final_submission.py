# fix_submission_final.py
import pandas as pd

# Load the original test file
test_df = pd.read_parquet('Task_B/test.parquet')

print(f"Test file columns: {test_df.columns.tolist()}")
print(f"Test file shape: {test_df.shape}")

# Check what ID column exists
if 'ID' in test_df.columns:
    test_ids = test_df['ID'].values
    print(f"Found ID column")
elif 'id' in test_df.columns:
    test_ids = test_df['id'].values
    print(f"Found id column")
else:
    # Use index as IDs
    test_ids = test_df.index.values
    print(f"Using index as IDs")

print(f"First 10 IDs from test: {test_ids[:10]}")

# Load your predictions (they should be in the same order as test file)
pred_df = pd.read_csv('submission.csv')

print(f"\nPredictions shape: {pred_df.shape}")
print(f"Predictions columns: {pred_df.columns.tolist()}")

# Create correct submission with original test IDs
submission = pd.DataFrame({
    'ID': test_ids,
    'label': pred_df['label'].values  # Your predictions in order
})

# Save
submission.to_csv('submission_kaggle.csv', index=False)

print(f"\n✓ Created submission_kaggle.csv")
print(f"✓ Total rows: {len(submission)}")
print(f"\nFirst 10 rows:")
print(submission.head(10))
print(f"\nMatches example format:")
print(f"  ID,label")
print(f"  {submission.iloc[0]['ID']},{submission.iloc[0]['label']}")