import os
import pandas as pd
from utils import download_images
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the dataset folder and image folders
DATASET_FOLDER = '../dataset/'
IMAGE_FOLDERS = {
    'train': '../images/train',
    'test': '../images/test',
    'sample_test': '../images/sample_test'
}

def download_dataset_images(csv_file, image_folder):
    df = pd.read_csv(os.path.join(DATASET_FOLDER, csv_file))
    image_links = df['image_link'].tolist()
    logging.info(f"Downloading {len(image_links)} images for {csv_file}")
    download_images(image_links, image_folder)

def main():
    for csv_name, image_folder in IMAGE_FOLDERS.items():
        try:
            csv_file = f"{csv_name}.csv"
            logging.info(f"Processing {csv_file}")
            download_dataset_images(csv_file, image_folder)
        except Exception as e:
            logging.error(f"Error downloading images for {csv_file}: {str(e)}")

if __name__ == "__main__":
    main()