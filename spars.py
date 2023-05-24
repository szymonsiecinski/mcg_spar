import argparse
import pickle
import math
from tqdm import tqdm
import os
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, savefig
from matplotlib import cm as CM
from ecgdetectors import Detectors
import hrv
import numpy as np
import scipy as sp
import re


def detect_heartbeats_in_ecg(ecg_signal, fs=800):
    detectors = Detectors(fs)
    return detectors.wqrs_detector(ecg_signal)


def detect_heartbeats_gcg(ecg_signal, gcg_signal, fs=800, delay=150):
    ecg_heartbeats=detect_heartbeats_in_ecg(ecg_signal, fs)
    gcg_heartbeats=list()

    for i in ecg_heartbeats:
        gcg_sample = gcg_signal[i: i+round(delay*fs/1000)]
        x=np.argmax(gcg_sample)
        gcg_heartbeats.append(i+x)

    return gcg_heartbeats


def calculate_hr(heartbeats, fs=800):
    hrvar = hrv.HRV(fs)
    return hrvar.HR(heartbeats)


def delayseq(x, delay_sec: float, fs: int):
    """
    x: input 1-D signal
    delay_sec: amount to shift signal [seconds]
    fs: sampling frequency [Hz]
    xs: time-shifted signal
    """

    assert x.ndim == 1, "only 1-D signals for now"

    delay_samples = delay_sec * fs
    delay_int = round(delay_samples)

    xs=np.roll(x, delay_int)

    return xs


def nextpow2(n: int) -> int:
    return 2 ** (int(n) - 1).bit_length()


def get_vw(raw_signal, heartbeats, fs=800):
    nn = np.diff(heartbeats)
    mean_nn = np.mean(nn)
    delay = mean_nn/3

    # delay a signal
    x = raw_signal
    y = delayseq(x, delay, fs)
    z = delayseq(x, 2*delay, fs)

    # create new v, and w vectors
    v = (x+y-2*z)/math.sqrt(6)
    w = (x-y)/math.sqrt(2)
    return (v, w)


def main(args):
    data_path = args.data_path
    fs = args.fs

    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    if args.interactive:
        i=3
        with open('{}/{}'.format(data_path, files[i]), 'rb') as handle:
            signal = pickle.load(handle)

        ecg = signal['dataframe']['ecg']
        gcg = signal['dataframe']['gcg_y']
        gcg_hb = detect_heartbeats_gcg(ecg, gcg, fs)

        v, w = get_vw(gcg, gcg_hb)

        figure()
        plot(v, w)
        xlabel('V')
        ylabel('W')
        title('Attractor reconstruction')
        
        # figure()
        # hexbin(v, w, C=None, gridsize=300, cmap=CM.jet, bins=None)
        # axis([v.min(), v.max(), w.min(), w.max()])
        # title('Heatmap')
        # xlabel('V')
        # ylabel('W')

        show()

    else:
        if not(os.path.exists("spars")):
            print("Creating spars directory")
            os.mkdir("spars")

        for i in tqdm(files, total=len(files)):
            subj_no = [int(s) for s in re.findall(r'\d+', i)]
            fname_spar_ecg = 'spar_ecg_sub{}'.format(subj_no[0])
            fname_spar_gcg = 'spar_gcg_sub{}'.format(subj_no[0])

            with open('{}/{}'.format(data_path, i), 'rb') as handle:
                signal = pickle.load(handle)

            #time = signal['time']
            ecg = signal['dataframe'].loc[:,'ecg']
            gcg = signal['dataframe'].loc[:,'gcg_y']
            
            ecg_hb = detect_heartbeats_in_ecg(ecg, fs)
            gcg_hb = detect_heartbeats_gcg(ecg, gcg, fs)

            v_ecg, w_ecg = get_vw(ecg, ecg_hb, fs)

            figure()
            plot(v_ecg, w_ecg)
            xlabel('V')
            ylabel('W')
            title('Attractor reconstruction for ECG {}'.format(i))
            savefig('spars/{}'.format(fname_spar_ecg), dpi=600, format='png')

            v_gcg, w_gcg = get_vw(gcg, gcg_hb, fs)

            figure()
            plot(v_gcg, w_gcg)
            xlabel('V')
            ylabel('W')
            title('Attractor reconstruction for GCG {}'.format(i))
            savefig('spars/{}'.format(fname_spar_gcg), dpi=600, format='png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs = '?', type = str, default = "signal_data/", help= "path to data files")
    parser.add_argument('--fs', nargs = '?', type = int, default = 800, help= "sampling frequency")
    parser.add_argument('--interactive', nargs='?', const=True, type=bool, default=False, help="interactive mode")
					
    args = parser.parse_args()
    main(args)