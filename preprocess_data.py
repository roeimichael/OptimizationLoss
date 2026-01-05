import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_dataset(input_path: str, output_train: str, output_test: str, target_column: str = 'Status'):
    print("="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Original dataset shape: {df.shape}")
    print(f"Original class distribution:\n{df[target_column].value_counts().sort_index()}")

    if any(col.startswith('Unnamed') for col in df.columns):
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
        print(f"\nRemoving unnamed index columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    if 'cost_matrix' in df.columns:
        print("Removing 'cost_matrix' column")
        df = df.drop(columns=['cost_matrix'])
    else:
        print("No 'cost_matrix' column found (skipping)")

    print(f"\nRemoving rows where {target_column} == 1")
    df_filtered = df[df[target_column] != 1].copy()
    print(f"Filtered dataset shape: {df_filtered.shape}")
    print(f"Removed {len(df) - len(df_filtered)} rows")
    print(f"Filtered class distribution:\n{df_filtered[target_column].value_counts().sort_index()}")

    print(f"\nSplitting into train/test with stratification on '{target_column}'")
    print("Test size: 20%, Random state: 42")
    train_df, test_df = train_test_split(
        df_filtered,
        test_size=0.2,
        random_state=42,
        stratify=df_filtered[target_column]
    )

    print(f"\nTrain dataset shape: {train_df.shape}")
    print(f"Train class distribution:\n{train_df[target_column].value_counts().sort_index()}")
    print(f"\nTest dataset shape: {test_df.shape}")
    print(f"Test class distribution:\n{test_df[target_column].value_counts().sort_index()}")

    train_df.to_csv(output_train, index=False)
    print(f"\n✓ Saved train dataset to: {output_train}")

    test_df.to_csv(output_test, index=False)
    print(f"✓ Saved test dataset to: {output_test}")

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess_data.py <input_csv_path>")
        print("\nExample:")
        print("  python preprocess_data.py data/raw_data.csv")
        print("\nThis will create:")
        print("  - data/train_dataset.csv")
        print("  - data/test_dataset.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_train = 'data/train_dataset.csv'
    output_test = 'data/test_dataset.csv'

    preprocess_dataset(input_path, output_train, output_test)
