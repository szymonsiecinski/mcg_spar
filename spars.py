import argparse
import pickle
import math
import scipy
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import os
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, savefig, close, contourf
from matplotlib import cm as CM
from ecgdetectors import Detectors
import hrv
import numpy as np
import scipy.signal as sp
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


def preprocess(raw_signal, fs=800, **kwargs):
    if not(kwargs.get('medfilt_window')):
        medfilt_window = 11
    else:
        medfilt_window = kwargs.get('medfilt_window')

    if not(kwargs.get('bpfilter_order')):
        bpfilter_order = 5
    else:
        bpfilter_order = kwargs.get('bpfilter_order')

    sos = sp.butter(bpfilter_order, (0.5, 40), btype='bandpass', fs=fs, output='sos')
    step1 = sp.sosfilt(sos, raw_signal)
    step2 = sp.medfilt(step1, medfilt_window)

    return step2


def calculate_features(v, w):
    v_min = np.min(v)
    v_max = np.max(v)
    w_min = np.min(w)
    w_max = np.max(w)
    v_avg = np.mean(v)
    w_avg = np.mean(w)
    v_sd = np.std(v)
    w_sd = np.std(w)
    v_median = np.median(v)
    w_median = np.median(w)
    hammdist_vw = scipy.spatial.distance.hamming(v, w)
    return {'v_min': v_min, 'v_max': v_max,
            'w_min': w_min, 'w_max': w_max,
            'v_avg': v_avg, 'w_avg': w_avg,
            'v_sd': v_sd, 'w_sd': w_sd,
            'v_median': v_median, 'w_median': w_median,
            'hamming_dist': hammdist_vw
            }


def save_features(features, filename, path):
        with open('{}/{}.pickle'.format(path, filename), 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(args):
    data_path = args.data_path
    fs = args.fs
    target_dir = args.target_dir

    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    if args.interactive:
        i = int(input('Enter the number of file (0-{}): '.format(len(files)-1)))
    
        if i > len(files)-1 and i < 0:
            print("Wrong number of file")
            return

        with open('{}/{}'.format(data_path, files[i]), 'rb') as handle:
            signal = pickle.load(handle)

        ecg = signal['ecg']
        gcg = signal['gcg_y']

        if args.preprocess:
            ecg = preprocess(ecg)
            gcg = preprocess(gcg)

        gcg_hb = detect_heartbeats_gcg(ecg, gcg, fs)

        v, w = get_vw(gcg, gcg_hb)

        figure()
        plot(v, w)
        xlabel('V')
        ylabel('W')
        title('Attractor reconstruction')
        show()

    else:
        if not(os.path.exists(target_dir)):
            print("Creating {}} directory".format(target_dir))
            os.mkdir(target_dir)

        for i in tqdm(files, total=len(files)):
            subj_no = [int(s) for s in re.findall(r'\d+', i)]
            fname_spar_ecg = 'spar_ecg_sub{}'.format(subj_no[0])
            fname_spar_scg = 'spar_scg_sub{}'.format(subj_no[0])
            fname_spar_gcg = 'spar_gcg_sub{}'.format(subj_no[0])

            with open('{}/{}'.format(data_path, i), 'rb') as handle:
                signal = pickle.load(handle)

            #time = signal['time']
            ecg = signal['ecg']
            scg = signal['scg_z']
            gcg = signal['gcg_y']
            
            if args.preprocess:
                ecg = preprocess(ecg)
                scg = preprocess(scg)
                gcg = preprocess(gcg)

            ecg_hb = detect_heartbeats_in_ecg(ecg, fs)
            scg_hb = detect_heartbeats_gcg(ecg, scg, fs)
            gcg_hb = detect_heartbeats_gcg(ecg, gcg, fs)

            v_ecg, w_ecg = get_vw(ecg, ecg_hb, fs)
            v_scg, w_scg = get_vw(scg, scg_hb, fs)
            v_gcg, w_gcg = get_vw(gcg, gcg_hb, fs)

            figure()
            plot(v_ecg, w_ecg)
            xlabel('V')
            ylabel('W')
            title('Attractor reconstruction for ECG {}'.format(i))
            savefig('{}/{}.png'.format(target_dir, fname_spar_ecg), dpi=150, format='png')
            close()

            figure()
            plot(v_scg, w_scg)
            xlabel('V')
            ylabel('W')
            title('Attractor reconstruction for SCG {}'.format(i))
            savefig('{}/{}.png'.format(target_dir, fname_spar_scg), dpi=150, format='png')
            close()

            figure()
            plot(v_gcg, w_gcg)
            xlabel('V')
            ylabel('W')
            title('Attractor reconstruction for GCG {}'.format(i))
            savefig('{}/{}.png'.format(target_dir, fname_spar_gcg), dpi=150, format='png')
            close()

            features = calculate_features(v_ecg, w_ecg)
            save_features(features, 'spar_ecg_features_sub{}'.format(subj_no[0]), target_dir)

            features = calculate_features(v_gcg, w_gcg)
            save_features(features, 'spar_gcg_features_sub{}'.format(subj_no[0]), target_dir)

            features = calculate_features(v_scg, w_scg)
            save_features(features, 'spar_scg_features_sub{}'.format(subj_no[0]), target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs = '?', type = str, default = "signal_data/", help= "path to data files")
    parser.add_argument('--target_dir', nargs = '?', type = str, default = "spars/", help= "path to save output files")
    parser.add_argument('--fs', nargs = '?', type = int, default = 800, help= "sampling frequency")
    parser.add_argument('--interactive', nargs='?', const=True, type=bool, default=False, help="interactive mode")
    parser.add_argument('--preprocess', nargs='?', type=bool, const=True, default=False, help='Preprocess raw signals')
					
    args = parser.parse_args()
    main(args)