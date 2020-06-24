import pandas as pd
import os

# Path relevant paths
LABELS_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/data", "labels_with_quality.csv")
DATA_PATH = "C:/Users/mgoldwater/Desktop/WHOI Storage/data"
DATA_DESCRIPTION_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/SAMBAY", "metadata/AllCalls_multichannel_2.txt")

# Read in the dataframe
labels = pd.read_csv(LABELS_PATH)

# Get the original data
df = pd.read_csv(DATA_DESCRIPTION_PATH, sep='\t')
df.columns = ['Selection',
                   'View',
                   'Channel',
                   'BeginTimes',
                   'EndTimes',
                   'LowFreq',
                   'HighFreq',
                   'BeginDateTime',
                   'DeltaFreq',
                   'DeltaTime',
                   'CenterFreq',
                   'PeakFreq',
                   'BasicCat',
                   'Quality',
                   'Localization']

# Filter out only the gunshots
df = df.query("BasicCat == '3'").reset_index()

# Append quality column to my data
labels['Channel'] = df['Channel']

# Save new labels file
labels.to_csv(os.path.join(DATA_PATH, "labels_with_quality_and_channel.csv"))
