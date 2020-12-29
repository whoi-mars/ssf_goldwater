# ssf_goldwater

## Overview

This repository contains work done as a part of the **2020 WHOI Summer Student Fellowship (SSF)**. The work done over the
course of the summer includes the following:
* Extracted gunshots (impulse calls) of Southern right whales from the audio files in 
  `SAMBAY/acounstic data/multichannel_wav_1h/`
* Labeled each of the gunshots as either *containing at least two modes* (427 images) and *containing less than two 
  modes* (620 images)
* Extracted call-length samples from the data where no call is present in order to have a group of images which
  represents a *noise* class of data
* Calculated spectrograms from each of the examples and architecting and training a convolutional neural network (CNN)
  to differentiate between the three classes of labeled data
* Performed hard negative mining for the *containing at least two modes* class after initial testing with the network
  
## Scanning Function

The function used to apply the trained model is in `utils/scanData.py` and is called `scan_audiofile`. It parses an 
audio file and saves indices where the CNN has labeled an audio snippet as a gunshot having at least two modes in a 
CSV file as well as the name of the audio file and the CNN's confidence in its label. If the same CSV file is used for 
multiple calls of the function, identified calls will be appended to the end of the file. The function takes 8 parameters
which are described below.

Parameter | Description | Required?
------------ | ------------- | -------------
data_path | File path to the directory where audio files of interest are stored | Yes
write_path | File path to the directory to where the output CSV file should be written | Yes
channel | The channel of the audio file to analyze | Yes
log_name | Name to give to the output CSV file (DO NOT INCLUDE ".csv") | No (default is "results.csv"
batch_size | The number of spectrograms to be simultaneously classified by the model | No (default 50)
batches | The number of batches to process from the provided audio file | No (will process the whole file if no argument provided)
save_spects | Boolean flag to enable saving reference images of spectrograms identified to be a gunshot call with at least two modes | No (default False)

### How To Use

There is no need to write any additional code to apply the classifier. One only needs to run the Python script `scanFiles.py`
and provide the appropriate inputs via flags as specified in the help message below:

```
usage: scanFiles.py [-h] [-ln LOG_NAME] [-s BATCH_SIZE] [-b BATCHES] [-i] -dp DATA_PATH -wp WRITE_PATH -c CHANNEL

optional arguments:
  -h, --help            show this help message and exit
  -ln LOG_NAME, --log_name LOG_NAME
                        Name of CSV log_file
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of spectrograms to be simultaneously classified
                        by the network. Default is 500
  -b BATCHES, --batches BATCHES
                        Number of batches to process from the audio file.
                        Default will process the whole file
  -i, --images          Save identified multi-modal spectrograms

required arguments:
  -dp DATA_PATH, --data_path DATA_PATH
                        Path to directory with .wav files to scan
  -wp WRITE_PATH, --write_path WRITE_PATH
                        Path to directory where results CSV and images are
                        stored
  -c CHANNEL, --channel CHANNEL
                        Audio channel to scan
```

## Development

Development of the CNN is on the `dev` branch of the repository in the file `dispersionID.py` and a structure for testing the CNN on data using the 
`scan_audofile` function is in the file `applyModel.py`. Both files are well-commented and were developed in PyCharm using the 
Cell Mode plugin which provides a Jupyter-Notebook-like coding environment.