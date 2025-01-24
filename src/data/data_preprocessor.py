import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.data = None
        self.train_data = None
        self.val_data = None

    def load_data(self, train_path: str, dev_path: str):
        """
        Load and merge train and dev datasets.
        """
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)
        self.data = pd.concat([train_df, dev_df], ignore_index=True)
        print("Data loaded successfully. Shape:", self.data.shape)

    def clean_data(self):
        """
        Clean the dataset by removing duplicates and handling missing values.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first using load_data().")
        # Remove duplicates
        self.data = self.data.drop_duplicates(subset=['fname', 'dialouge'])
        # Drop rows with missing values
        self.data = self.data.dropna()
        print("Data cleaned. Shape:", self.data.shape)

    def split_data(self, stratify_column: str, test_size: float = 0.2, random_state: int = 42):
        """
        Split the dataset into train and validation sets.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load and clean data first.")
        self.train_data, self.val_data = train_test_split(
            self.data,
            test_size=test_size,
            stratify=self.data[stratify_column],
            random_state=random_state
        )
        print(f"Data split into train ({len(self.train_data)}) and validation ({len(self.val_data)}) sets.")

    def save_data(self, train_path: str, val_path: str):
        """
        Save train and validation datasets to CSV.
        """
        if self.train_data is None or self.val_data is None:
            raise ValueError("Data not split. Please split data first using split_data().")
        self.train_data.to_csv(train_path, index=False)
        self.val_data.to_csv(val_path, index=False)
        print(f"Train data saved to {train_path}.")
        print(f"Validation data saved to {val_path}.")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.load_data("data/train.csv", "data/dev.csv")
    processor.clean_data()
    processor.split_data("topic")
    processor.save_data("data/new_train.csv", "data/new_validation.csv")