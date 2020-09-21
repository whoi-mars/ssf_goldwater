# ssf_goldwater

## Overview

This repository contains work done as a part of the **2020 WHOI Summer Student Fellowship (SSF)**. The work done over the
course of the summer includes the following:
* Extracting gunshots (impulse calls) of Southern right whales from the audio files in 
  `SAMBAY/acounstic data/multichannel_wav_1h/`
* Labeling each of the gunshots as either *containing at least two modes* (427 images) and *containing less than two 
  modes* (620 images)
* Extracting call-length samples from the data where no call is present in order to have a group of images which
  represents a *noise* class of data
* Calculating spectrograms from each of the examples and architecting and training a convolutional neural network (CNN)
  to differentiate between the three classes of labeled data
* Performing hard negative mining for the *containing at least two modes* class after initial testing with the network
  
## How to Use

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
log_name | Name to give to the output CSV file | Yes
batch_size | The number of spectrograms to be simultaneously classified by the model | No (default 50)
batches | The number of batches to process from the provided audio file | No (will process the whole file if no argument provided)
save_spects | Boolean flag to enable saving reference images of spectrograms identified to be a gunshot call with at least two modes | No (default False)

### Steps To Use

1. Clone the ssf_goldwater GitHub repository onto your computer.
2. Create a new Python script and import the function `scan_audiofile` which is used to scan new audio data. If the new Python file
is in the same directory as the cloned repository, this can be done by typing the following at the top of the new script: 
    
    ```from ssf_goldwater.utils.scanData import scan_audiofile```
3. Look at the required parameters above and make sure that you are able to fill those in which you need for your application.
Below is a sample call of the function:

    ```scan_audiofile("C:/Users/jdoe/Desktop/audiofiles", "C:/Users/jdoe/Desktop/outputfiles", 1, "Southern_right_whale_gunshots")```

## Development

Development of the CNN is in the file `dispersionID.py` and a structure for testing the CNN on data using the 
`scan_audofile` function is in the file `applyModel.py`. Both files are well-commented and were developed in PyCharm using the 
Cell Mode plugin which provides a Jupyter-Notebook-like coding environment.