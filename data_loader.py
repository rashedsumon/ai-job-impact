import os
import glob
import pandas as pd
import kagglehub

def download_and_load_dataset():
    """
    Downloads the 'sarcasmos/ai-society' dataset using kagglehub
    and loads the main CSV file into a pandas DataFrame.
    """
    print("Fetching dataset from Kaggle...")
    # Downloads the latest version to a local cache directory
    path = kagglehub.dataset_download("sarcasmos/ai-society")
    print(f"Path to dataset files: {path}")
    
    # Locate any CSV file inside the downloaded directory
    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset.")
    
    # Pick the first matching CSV file found
    target_csv = csv_files[0]
    print(f"Loading data from: {target_csv}")
    
    df = pd.read_csv(target_csv)
    return df

if __name__ == "__main__":
    # Test execution
    df = download_and_load_dataset()
    print("Dataset Sample Head:")
    print(df.head(2))