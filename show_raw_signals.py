import pandas
import argparse
import pickle
#from tqdm import tqdm
import os
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, savefig


def main(args):
    data_path = args.data_path
    fs = args.fs
    #signal = list()
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    i=3
    with open('saved_data/{}'.format(files[i]), 'rb') as handle:
        signal = pickle.load(handle)

    figure()
    plot(signal['time'], signal['dataframe'].loc[:,'ecg'])
    xlabel('Time [s]')
    ylabel('Amplitude')
    title('ECG signal')
    #savefig('')

    figure()
    plot(signal['time'], signal['dataframe'].loc[:,'gcg_y'])
    xlabel('Time [s]')
    ylabel('Amplitude')
    title('GCG signal')
    show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs = '?', type = str, default = "saved_data/", help= "path to data files")
    parser.add_argument('--fs', nargs = '?', type = int, default = 800, help= "sampling frequency")
					
    args = parser.parse_args()
    main(args)