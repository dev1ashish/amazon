import re
import constants
import os
import requests
import pandas as pd
import multiprocessing
import time
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib
from PIL import Image
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

def common_mistake(unit):
    if unit in constants.allowed_units:
        return unit
    if unit.replace('ter', 'tre') in constants.allowed_units:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in constants.allowed_units:
        return unit.replace('feet', 'foot')
    return unit

def parse_string(s):
    s_stripped = "" if s==None or str(s)=='nan' else s.strip()
    if s_stripped == "":
        return None, None
    pattern = re.compile(r'^-?\d+(.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError("Invalid format in {}".format(s))
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in constants.allowed_units:
        raise ValueError("Invalid unit [{}] found in {}. Allowed units: {}".format(
            unit, s, constants.allowed_units))
    return number, unit

def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        return

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except:
            time.sleep(delay)

    create_placeholder_image(image_save_path) #Create a black placeholder image for invalid links/images

def download_images(image_links, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        download_image_partial = partial(
            download_image, save_folder=download_folder, retries=3, delay=3)

        with multiprocessing.Pool(64) as pool:
            list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
            pool.close()
            pool.join()
    else:
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)

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