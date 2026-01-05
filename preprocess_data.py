import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_dataset(input_path: str, output_train: str, output_test: str, target_column: str = 'Target'):
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Original dataset shape: {df.shape}")
    print(f"Original class distribution:\n{df[target_column].value_counts().sort_index()}")

    # Drop unnamed columns (like 'Unnamed: 0')
    if any(col.startswith('Unnamed') for col in df.columns):
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
        print(f"\nRemoving unnamed index columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    # Drop the cost_matrix column if it exists
    if 'cost_matrix' in df.columns:
        print("Removing 'cost_matrix' column")
        df = df.drop(columns=['cost_matrix'])
    else:
        print("No 'cost_matrix' column found (skipping)")

    # ---------------------------------------------------------
    # CHANGE 1: Filter rows where 'Course' == 1
    # ---------------------------------------------------------
    if 'Course' in df.columns:
        print(f"\nRemoving rows where Course == 1")
        initial_len = len(df)
        df_filtered = df[df['Course'] != 1].copy()
        print(f"Filtered dataset shape: {df_filtered.shape}")
        print(f"Removed {initial_len - len(df_filtered)} rows")
    else:
        print(f"\nWarning: 'Course' column not found. Skipping filtering.")
        df_filtered = df.copy()

    # ---------------------------------------------------------
    # CHANGE 2: Convert Target to numerical
    # ---------------------------------------------------------
    print(f"\nEncoding target column '{target_column}' to numerical values")
    le = LabelEncoder()
    df_filtered[target_column] = le.fit_transform(df_filtered[target_column])

    # Print the mapping for your reference
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Target mapping applied: {mapping}")
    print(f"Filtered class distribution (numerical):\n{df_filtered[target_column].value_counts().sort_index()}")

    # Split into train and test
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

    # Save outputs
    train_df.to_csv(output_train, index=False)
    print(f"\n✓ Saved train dataset to: {output_train}")

    test_df.to_csv(output_test, index=False)
    print(f"✓ Saved test dataset to: {output_test}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Update paths as needed
    input_path = './data/dataset.csv'
    output_train = './data/dataset_train.csv'
    output_test = './data/dataset_test.csv'

    # Note: Column name is 'Target', not 'Status' based on your file
    preprocess_dataset(input_path, output_train, output_test, target_column="Target")