import argparse
import os

# Function inputs
DATA_PATH = "" # Path to directory with audiofiles
WRITE_PATH = "" # Path to directory to store output csv and images
CHANNEL = 1 # Audio channel to scan
LOG_NAME = 'results_file' # Name of output csv
BATCH_SIZE = 500 # Number of spectrograms to simultaneously process
BATCHES = None # Number of batches to process (None will process whole file)
IMAGES = False # Set to True is you wish to save images of multi-modal spectrograms

# Get list of audiofiles to scan
audio_files = os.listdir(DATA_PATH)

# Import pre-trained classifier
from utils.scanData import scan_audiofile

# Scan audio
for file in audio_files:
    FILE_PATH = os.path.join(DATA_PATH, file)
    scan_audiofile(data_path=FILE_PATH, write_path=WRITE_PATH, channel=CHANNEL, log_name=LOG_NAME,
                   batch_size=BATCH_SIZE, batches=BATCHES, save_spects=IMAGES)
