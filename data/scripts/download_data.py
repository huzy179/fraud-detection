"""
Script to download Credit Card Fraud Detection dataset from Kaggle.
Alternative if no Kaggle API key: use sklearn's make_classification or
download from alternative sources.
"""

import os
import zipfile
import urllib.request

DATASET_URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "raw", "creditcard.csv")


def download_kaggle():
    """Download via Kaggle API."""
    os.system(
        "kaggle datasets download -d mlg-ulb/creditcardfraud -p "
        + os.path.join(os.path.dirname(__file__), "..", "raw")
    )
    zip_path = os.path.join(os.path.dirname(__file__), "..", "raw", "creditcardfraud.zip")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(os.path.join(os.path.dirname(__file__), "..", "raw"))


def download_direct():
    """Direct download as fallback."""
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "raw"), exist_ok=True)
    print(f"Downloading dataset to {OUTPUT_PATH}...")
    urllib.request.urlretrieve(DATASET_URL, OUTPUT_PATH)
    print("Done!")


if __name__ == "__main__":
    try:
        import kaggle  # noqa
        download_kaggle()
    except ImportError:
        download_direct()
