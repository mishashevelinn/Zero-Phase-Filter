import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


class ZeroPhaseFilter:
    def __init__(self,
                 fs,
                 file_path,
                 number_of_channels,
                 num_ADC_bits,
                 voltage_resolution
                 ):
        """

        :param fs: sampling frequency
        :param file_path: path to raw data file
        :param number_of_channels: number of channels in signal
        :param num_ADC_bits: number of bits in A/D converter
        :param voltage_resolution: resolution of A/D converter
        """
        self.fs = fs
        self.file_path = file_path
        self.number_of_channels = number_of_channels
        self.filtered_channels = np.zeros((number_of_channels))
        self.voltage_resolution = voltage_resolution
        self.num_ADC_bits = num_ADC_bits

    def read(self):
        """

        :return: numpy array of integers
        """
        file = None
        try:
            file = np.fromfile(self.file_path, dtype=np.uint16)
        except Exception as e:
            print("could not open the file")
        return file

    def quantize(self):
        """

        :return: quantized data matrix
        each row represents a channels
        """
        data = self.read()
        data = np.reshape(data, (self.number_of_channels, -1),
                          order='F')
        data = np.multiply(self.voltage_resolution,
                           (data - np.float_power(2, self.num_ADC_bits - 1)))
        return data

    def frame(self):
        """

        :return: data matrix to pandas DataFrame
        """
        data = self.quantize()
        df = pd.DataFrame(data=data, index=["channel" + str(i) for i in range(1, data.shape[0] + 1)])
        return df

    def filter(self, low_cut, high_cut, order):
        """
        band pass range from low_cut to high_cut
        :param order: order of filter
        :return: array of filtered channels
        """
        nyq = 0.5 * self.fs
        low = low_cut / nyq
        high = high_cut / nyq
        b, a = butter(order, [low, high], 'bandpass', analog=False, output='ba')
        channels = self.frame()
        self.filtered_channels = np.zeros((number_of_channels, channels.shape[1]))
        for i in range(self.number_of_channels):
            self.filtered_channels[i] = filtfilt(b, a, channels.iloc[i])
        return self.filtered_channels

    def plot(self, a=0, b=3):
        """
        :param range: tuple - plots in a range from a*fs to b*fs
        :return:
        """
        nn = np.arange(0, self.frame().shape[1])
        fig, ax = plt.subplots(self.number_of_channels, figsize=(10, 15))
        for i in range(self.number_of_channels):
            ax[i].set_ylabel('Volt')
            ax[i].set_xlabel('time')
            ax[i].set_title("channel " + str(i + 1))
            ax[i].plot(nn[a * fs: b * fs], self.filtered_channels[i][a * fs: b * fs])
        plt.tight_layout()
        plt.show()


# usage example
low_cut = 20.0
high_cut = 30.0
fs = 4000
file_path = 'NEUR0000.DT8'
number_of_channels = 8
voltage_resolution = 4.12e-7
num_ADC_bits = 15
order = 2
z_filter = ZeroPhaseFilter(fs, file_path, number_of_channels, num_ADC_bits, voltage_resolution)
z_filter.quantize()
z_filter.filter(low_cut, high_cut, order)
z_filter.plot()
