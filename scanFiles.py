##

import argparse
import os

# Collect arguments
parser = argparse.ArgumentParser()
parser.add_argument("-ln", "--log_name", type=str, help="Name of CSV log_file")
parser.add_argument("-s", "--batch_size", type=int, help="Number of spectrograms to be simultaneously classified by the network. Default is 500")
parser.add_argument("-b", "--batches", type=int, help="Number of batches to process from the audio file. Default will process the whole file")
parser.add_argument("-i", "--images", help="Save identified multi-modal spectrograms", action="store_true")
required = parser.add_argument_group('required arguments')
required.add_argument("-dp", "--data_path", type=str, help="Path to directory with .wav files to scan", required=True)
required.add_argument("-wp", "--write_path", type=str, help="Path to directory where results CSV and images are stored", required=True)
required.add_argument("-c", "--channel", type=int, help="Audio channel to scan", required=True)

# Parse arguments
args = parser.parse_args()
log_name = 'results' if args.log_name is None else args.log_name
batch_size = 500 if args.batch_size is None else args.batch_size

# Get audiofile names
DATA_PATH = args.data_path
audio_files = os.listdir(DATA_PATH)

# Import pre-trained classifier
from utils.scanData import scan_audiofile

# Scan audio
for file in audio_files:
    FILE_PATH = os.path.join(DATA_PATH, file)
    scan_audiofile(data_path=FILE_PATH, write_path=args.write_path, channel=args.channel, log_name=log_name,
                   batch_size=batch_size, batches=args.batches, save_spects=args.images)
