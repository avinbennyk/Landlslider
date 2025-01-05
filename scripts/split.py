import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to split the dataset
def split_dataset(file_path, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    Args:
        file_path (str): Path to the preprocessed dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    print("Preprocessed data loaded successfully.")

    # Split the dataset into features (X) and target (y)
    features = ["Slope", "Aspect", "Curvature", "Precipitation", "NDVI", "NDWI", "Elevation"]
    target = "Landslide"
    X = data[features]
    y = data[target]

    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Directory to save the files
    output_dir = "/Users/avinbennyk/Desktop/Landslidepro/Dataset/"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Save training data
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    print("Training data saved to 'train_data.csv'.")

    # Save testing data
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
    print("Testing data saved to 'test_data.csv'.")

# Main function
if __name__ == "__main__":
    # File path to the preprocessed dataset
    file_path = "/Users/avinbennyk/Desktop/Landslidepro/Dataset/preprocessed_data.csv"  # Reference the correct file name

    # Split the dataset
    print("Splitting the dataset into training and testing sets...")
    split_dataset(file_path)
    print("Dataset splitting completed.")
