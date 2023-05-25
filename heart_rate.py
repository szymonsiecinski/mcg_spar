import argparse
import pickle
from tqdm import tqdm
import os
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, legend, show
from ecgdetectors import Detectors
import hrv
import numpy as np
import re


def detect_heartbeats(ecg_signal, gcg_signal, fs=800, delay=150):
    detectors = Detectors(fs)
    
    ecg_heartbeats=detectors.wqrs_detector(ecg_signal)
    gcg_heartbeats=list()

    for i in ecg_heartbeats:
        gcg_sample = gcg_signal[i: i+round(delay*fs/1000)]
        x=np.argmax(gcg_sample)
        gcg_heartbeats.append(i+x)

    return (ecg_heartbeats, gcg_heartbeats)


def calculate_hr(ecg_hbeats, gcg_hbeats, fs=800):
    hrvar = hrv.HRV(fs)
    ecg_hr = hrvar.HR(ecg_hbeats)
    gcg_hr = hrvar.HR(gcg_hbeats)

    return (ecg_hr, gcg_hr)


def save_heartbeats(filename, time, ecg_heartbeats, gcg_heartbeats, fs=800):
    heartbeats = {'time': time, 'ecg': ecg_heartbeats, 'gcg': gcg_heartbeats, 'fs': fs}

    if not(os.path.exists("heartbeats")):
        print ("Creating Heartbeats Path")
        os.mkdir("heartbeats")

    with open('heartbeats/{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(heartbeats, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_hr(filename, time, ecg_hr, gcg_hr, fs=800):
    hr = {'time': time, 'ecg_hr': ecg_hr, 'gcg_hr': gcg_hr, 'fs': fs}

    if not(os.path.exists("hr")):
        print("Creating HR folder")
        os.mkdir("hr")

    with open('hr/{}.pickle'.format(filename), 'wb') as f:
        pickle.dump(hr, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(args):
    data_path = args.data_path
    fs = args.fs
    #signal = list()
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    if(args.interactive):
        i = int(input('Enter the number of file (0-{}): '.format(len(files)-1)))
        with open('{}/{}'.format(data_path, files[i]), 'rb') as handle:
            signal = pickle.load(handle)

        ecg = signal['dataframe'].loc[:,'ecg']
        gcg = signal['dataframe'].loc[:,'gcg_y']
        ecg_hb, gcg_hb = detect_heartbeats(ecg, gcg, fs)

        ecg_hr, gcg_hr = calculate_hr(ecg_hb, gcg_hb, fs)

        figure()
        plot(signal['time'], ecg)
        title('ECG')
        ylabel('Amplitude [a.u.]')
        xlabel('Time [s]')
        plot(signal['time'][ecg_hb], ecg[ecg_hb], '*')

        figure()
        plot(signal['time'], gcg)
        title('GCG')
        ylabel('Amplitude [a.u.]')
        xlabel('Time [s]')
        plot(signal['time'][gcg_hb], gcg[gcg_hb], '*')

        figure()
        plot(ecg_hr, label='ECG')
        plot(gcg_hr, label='GCG')
        title('Heart rate in ECG and GCG')
        ylabel('HR [bpm]')
        xlabel('Time [s]')
        legend()

        show()

    else:
        for i in tqdm(files, total=len(files)):
            subj_no = [int(s) for s in re.findall(r'\d+', i)]
            fname_hb = 'heartbeats_sub{}'.format(subj_no[0])
            fname_hr = 'hr_sub{}'.format(subj_no[0])

            with open('{}/{}'.format(data_path, i), 'rb') as handle:
                signal = pickle.load(handle)

            time = signal['time']
            ecg = signal['dataframe'].loc[:,'ecg']
            gcg = signal['dataframe'].loc[:,'gcg_y']
            
            ecg_hb, gcg_hb = detect_heartbeats(ecg, gcg, fs)
            save_heartbeats(fname_hb, time, ecg_hb, gcg_hb, fs)

            ecg_hr, gcg_hr = calculate_hr(ecg_hb, gcg_hb, fs)
            save_hr(fname_hr, ecg_hr, gcg_hr, fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs = '?', type = str, default = "signal_data/", help= "path to data files")
    parser.add_argument('--fs', nargs = '?', type = int, default = 800, help= "sampling frequency")
    parser.add_argument('--interactive', nargs='?', const=True, type=bool, default=False, help="interactive mode")
					
    args = parser.parse_args()
    main(args)