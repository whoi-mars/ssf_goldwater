import pandas as pd
import os


def add_column(labels_path, destination_path, data_description_path, column, query):

    """
    This function takes a column from one data frame (at data_description_path) and adds it to another one
    (at labels_path). It will also query the data frame from which the column is coming.

    :param labels_path: Path to labels csv file to add column to
    :param destination_path: Path to the folder in which the pickled data will go
    :param data_description_path: Path to the original table where the column is coming from
    :param column: name of column to be added
    :param query: query to filter the original data description table to match the labels table.
    """

    # Read in the dataframe
    labels = pd.read_csv(labels_path)

    # Get the original data
    df = pd.read_csv(data_description_path, sep='\t')
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
    df = df.query(query).reset_index()

    # Append quality column to my data
    labels[column] = df[column]

    # Save new labels file
    labels.to_csv(os.path.join(destination_path, "labels_wfdsafdaith_{}.csv".format(column)))


if __name__ == "__main__":

    # Path relevant paths
    LABELS_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/data", "labels_with_quality.csv")
    DESTINATION_PATH = "C:/Users/mgoldwater/Desktop/WHOI Storage/data"
    DATA_DESCRIPTION_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/SAMBAY",
                                         "metadata/AllCalls_multichannel_2.txt")

    # Add a column from main data description table to my labels
    add_column(LABELS_PATH, DESTINATION_PATH, DATA_DESCRIPTION_PATH, 'Channel', "BasicCat == '3'")
