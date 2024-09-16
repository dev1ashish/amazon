import os
import pandas as pd
from utils import download_images
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATASET_FOLDER = 'dataset/'
IMAGE_FOLDERS = {
    'train': 'images/train',
    'test': 'images/test',
    'sample_test': 'images/sample_test'
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def download_dataset_images(csv_file, image_folder):
    df = pd.read_csv(csv_file)
    image_links = df['image_link'].tolist()
    
    logging.info(f"Downloading images for {csv_file}")
    logging.info(f"Total images to download: {len(image_links)}")
    
    ensure_dir(image_folder)
    download_images(image_links, image_folder)
    
    logging.info(f"Finished downloading images for {csv_file}")

def main():
    for dataset, folder in IMAGE_FOLDERS.items():
        csv_file = os.path.join(DATASET_FOLDER, f"{dataset}.csv")
        if os.path.exists(csv_file):
            download_dataset_images(csv_file, folder)
        else:
            logging.warning(f"CSV file not found: {csv_file}")

if __name__ == "__main__":
    main()
    logging.info("Image download process completed.")