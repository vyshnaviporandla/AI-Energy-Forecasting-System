from src.data import load_dataset
from src.train import train

if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset ready: {len(df):,} hourly rows")
    print(df.head().to_string(index=False))
    train()
