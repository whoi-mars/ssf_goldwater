from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def countFN(X, start, end, csv_path):

    """
    :param X: List of images to be labeled as 1 (TN) or 0 (FN)
    :param start: Start index for the subset of images to be labeled
    :param end: End index for the subset of images to be labeled
    :param csv_path: Path to the CSV file where the labels will be stored
    """

    # Get the CSV file of labels
    calls_df = pd.read_csv(csv_path, index_col=[0])

    # Assume everything is a TN to start
    for row in range(start, end):
        calls_df.loc[row, "Label"] = 4
    calls_df.to_csv(csv_path)

    # Function to respond to clicking on a plot
    def onclick(event):

        global ix
        ix = event.xdata
        for j, ax in enumerate(fig.axes):
            if ax == event.inaxes:
                calls_df.loc[start + j, "Label"] = 3
                calls_df.to_csv(csv_path)
                print("row {} marked as FN".format(start + j))

    num_per_row = 10
    rows = np.ceil(len(X[start:end]) / num_per_row)

    plt.ion()
    fig = plt.figure(figsize=(5, 5))
    for i, image in enumerate(X[start:end]):
        plt.subplot(rows, num_per_row, i + 1)
        plt.axis("off")
        plt.imshow(image)

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
