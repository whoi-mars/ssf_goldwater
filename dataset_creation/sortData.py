import os
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import utils.spect as spect


class ImageSortGUI:

    """
    Simple user interface to create spectrograms from data and a table of time stamps to crops at
    and then sort the data and document its labels in a CSV file.

    Instructions:
        - To start, run the file and click on the buttons to sort the data into its labels.
        - Change the window length, buffer time, or origin time-shift in the textboxes and hit ENTER to update the spectrogram
        - When you are done close the window
    """

    def __init__(self, master, labels, data_path, write_path):

        """
        Initialize the GUI

        :param master: The parent window
        :param labels: A list of labels used to sort the spectrograms
        :param data_path: Path to the data
        :param write_path: Path to directory where sorted spectrograms will go
        """

        # ----------- Main Window Setup -----------------------#

        # To use master window in functions
        self.master = master
        self.master.bind('<Return>', self.update_spectrogram)

        # Done state variable
        self.done = False

        # Extract frame so we can draw on it
        frame = tk.Frame(master)

        # Initialize pack
        frame.pack()

        # Current figure
        self.fig = plt.figure()

        # Variable to store spectrogram
        self.spectrogram = None

        # Initialize canvas
        self.canvas = None

        # ----------- Main Window Setup -----------------------#

        # ----------- Connect to and Create Dataframes and Destination Folders --------------------- #

        # Save labels and various paths
        self.labels = labels
        self.data_path = os.path.join(data_path, "acoustic data", "multichannel_wav_1h")
        self.data_table_path = os.path.join(data_path, "metadata", "AllCalls_multichannel_2.txt")
        self.write_path = write_path

        # Setup file structure
        self.create_directories()

        # Read in .txt file which describes the data as a pandas dataframe
        self.df = pd.read_csv(self.data_table_path, sep='\t')
        self.df.columns = ['Selection',
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
        self.df = self.df.query("BasicCat == '3'").reset_index()

        # Labels for CSV file
        self.labels_df_cols = ['File', 'Label', 'WindowNumber', 'Buffer', 'StartTime', 'EndTime', 'TimeShift']

        # Start row index at zero or pick up where last left off if 'labels.csv' exists
        if os.path.exists(os.path.join(self.write_path, "labels.csv")):
            self.labels_df = pd.read_csv(os.path.join(self.write_path, "labels.csv")).iloc[:, 1:]
            self.labels_df.columns = self.labels_df_cols
            self.index = len(self.labels_df)

            # Check if already done
            if self.index == len(self.df):
                print("You're already done!")
                self.stop()
        else:
            self.index = 0
            self.labels_df = pd.DataFrame(columns=self.labels_df_cols)

        # ----------- Connect to and Create Dataframes and Destination Folders --------------------- #

        # ---------------------------Create Buttons and Labels ------------------------------------- #

        # Create the buttons
        self.buttons = []
        for label in labels:
            self.buttons.append(tk.Button(frame,
                                          text=label,
                                          width=10,
                                          height=1,
                                          command=lambda l=label: self.move_image(l)))

        # back_time_button = tk.Button(frame, text="T-", width=5, height=1)

        # Place buttons in the window
        for ll, button in enumerate(self.buttons):
            button.grid(row=0, column=ll, sticky='we')

        # Variable for time shift input
        self.tshift = tk.DoubleVar()
        self.tshift.set(0.0)

        # Current start and stop time
        self.start_time = 0.0
        self.stop_time = 0.0

        # Label and entry field for time shift
        label_tshift = tk.Label(self.master, text="T-Shift")
        label_tshift.pack(pady=(20, 0))
        entry_tshift = tk.Entry(self.master, textvariable=self.tshift)
        entry_tshift.pack()

        # Create window variable
        self.window_N = tk.IntVar()
        self.window_N.set(31)

        # Create buffer variable
        self.buffer = tk.DoubleVar()
        self.buffer.set(0.1)

        # Store current file name
        self.corresponding_file = ""

        # Textbox and label for window variable
        label_window = tk.Label(self.master, text="Window")
        label_window.pack()
        text_entry_window = tk.Entry(self.master, textvariable=self.window_N)
        text_entry_window.pack()

        # Textbox and label for buffer variable
        label_buffer = tk.Label(self.master, text="Time Buffer")
        label_buffer.pack()
        text_entry_buffer = tk.Entry(self.master, textvariable=self.buffer)
        text_entry_buffer.pack()

        # --------------------------- Create Buttons and Labels ------------------------------------- #

        # --------------------------- Lead Initial Spectrogram -------------------------------------- #

        # Load the first spectrogram
        self.load_spectrogram(self.index)

        # Initialize canvas
        self.canvas = None

        # Set the first spectrogram
        self.set_spectrogram()

        # --------------------------- Lead Initial Spectrogram -------------------------------------- #

    def stop(self, event=None):

        """
        Quits the tkinter window.

        :param event: Contains the escape event object which executed the function.
        """

        self.master.quit()
        self.master.destroy()
        exit(1)

    def create_directories(self):

        """
        Create directories to sort the spectrograms associated with their respective labels.
        """

        for label in self.labels:
            data_folder_path = os.path.join(self.write_path, label)
            if not os.path.exists(data_folder_path):
                os.mkdir(data_folder_path)

    def load_spectrogram(self, index):

        """
        Created and saves a spectrogram according to the inputted index, and saves
        it in self.spectrogram.

        :param index: Row index in data table to turn into a spectrogram
        """

        # Get current row and the name of the corresponding file
        curr_row = self.df.loc[index]
        self.start_time, self.stop_time, self.corresponding_file = spect.parse_begin_date_and_time(curr_row)

        # Get and normalize the data from the file
        fs, samples_norm = spect.get_and_normalize_sound(os.path.join(self.data_path, self.corresponding_file))

        # Get the start and end time of the call, and add a 0.5 [s] buffer on either side.
        # Then, convert these times to indices in the vector
        starti, stopi = spect.range_to_indices(self.start_time - self.buffer.get() + self.tshift.get(),
                                               self.stop_time + self.buffer.get() + self.tshift.get(),
                                               fs)

        # Get the channel, subtracting one for indexing purposes
        channel = curr_row.Channel - 1

        # Crop the sound vector
        samples = samples_norm[starti:stopi, channel]

        # Calculate the stft
        time, freq, Zxx, fs = spect.my_stft(samples, fs, self.window_N.get())
        Zxx = Zxx[round(Zxx.shape[0] / 2):, :]
        freq = freq[:round(len(freq)/2)]
        spectro = np.abs(Zxx) ** 2

        # Clear the figure
        self.fig.clf()
        # get current axes
        ax = plt.gca()
        # Set ticks to be at pixel edge rather than center
        ax.set_xticks(np.linspace(-0.5, Zxx.shape[1] - 0.5, 9))
        ax.set_yticks(np.linspace(Zxx.shape[0] - 0.5, -0.5, 9))
        # Relabel the ticks
        ax.set_xticklabels(np.round(np.linspace(time[0], time[-1], 9), 3))
        ax.set_yticklabels(np.round(np.linspace(freq[0], freq[-1], 9), 2))
        # Plot image
        plt.imshow(10 * np.log10(spectro), aspect='auto')
        # Make sure labels are visible
        plt.subplots_adjust(left=0.15, bottom=0.1)
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")

        # Save the plot in spectrogram variable
        self.spectrogram = self.fig

    def set_spectrogram(self):

        """
        Displays spectrogram according to the inputted index in the window.
        """

        # Count index
        print("File: {}".format(self.corresponding_file))
        print("Selection: {} -- Spectogram: {}/{}".format(self.df.loc[self.index].Selection,
                                                          self.index + 1,
                                                          len(self.df)))

        # Display the spectrogram
        try:
            self.canvas.get_tk_widget().pack_forget()
        except AttributeError:
            pass

        if self.done:
            pass
        else:
            self.canvas = FigureCanvasTkAgg(self.spectrogram, master=self.master)
            self.canvas.get_tk_widget().pack(padx=(20, 20), pady=(20, 20))
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def move_image(self, l):

        """
        Updates the CSV file containing labels for each image and saves the spectrogram in its
        respective folder. The function then plots the next spectrogram on the GUI

        :param l: label of image to be saved
        """

        file_name = self.corresponding_file.split('.')[0] + '_selec_' + str(self.df.loc[self.index].Selection) + '.png'
        row = [file_name, self.labels.index(l), self.window_N.get(), self.buffer.get(), self.start_time, self.stop_time, self.tshift.get()]
        self.labels_df.loc[len(self.labels_df)] = row

        self.labels_df.to_csv(os.path.join(self.write_path, "labels.csv"))

        self.spectrogram.savefig(os.path.join(self.write_path, l, file_name))

        self.index += 1

        self.tshift.set(0.0)

        if self.index == len(self.df):
            self.done = True

        if not self.done:
            self.load_spectrogram(self.index)
        else:
            messagebox.showinfo("Spectrogram Sorter", "You're done!")

        self.set_spectrogram()

    def update_spectrogram(self, event):

        """
        Recreates and displays the spectrogram. To be used when the window length or buffer time are changed.

        :param event: Enter event which triggers this function
        """

        self.load_spectrogram(self.index)
        self.set_spectrogram()


if __name__ == "__main__":

    # Get paths for data and folder to sort into
    write_path = "C:/Users/mgoldwater/Desktop/WHOI Storage/data"
    data_path = "C:/Users/mgoldwater/Desktop/WHOI Storage/SAMBAY"

    # List of possible labels
    labels = ['no_dispersion', 'dispersion']

    # Create window and start up the app
    root = tk.Tk()
    root.title("Spectrogram Sorter")
    app = ImageSortGUI(root, labels, data_path, write_path)
    root.mainloop()
