import pandas
import argparse
import pickle
from tqdm import tqdm
import os
import re
import numpy
import math


def main(args):
    data_path = args.data_path
    target_dir = args.target
    fs = args.fs

    if not(os.path.exists(target_dir)):
        print ("Creating Saved Data Path")
        os.mkdir(target_dir)

    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    col_names = ['ecg', 'scg_x', 'scg_y', 'scg_z', 'gcg_x', 'gcg_y', 'gcg_z']

    for i in tqdm(files, total=len(files)):
        subj_no = [int(s) for s in re.findall(r'\d+', i)]
        
        if subj_no[0] < 4:
            df = pandas.read_csv(os.path.join(data_path, i), delim_whitespace=True, header=15, names=col_names)
        elif (subj_no[0] >= 4 and subj_no[0] < 7) or subj_no[0] == 2:
            df = pandas.read_csv(os.path.join(data_path, i), delim_whitespace=True, header=16, names=col_names)
        elif subj_no[0] > 7 and subj_no[0] < 10:
            df = pandas.read_csv(os.path.join(data_path, i), delim_whitespace=True, header=17, names=col_names)
        else:
            df = pandas.read_csv(os.path.join(data_path, i), delim_whitespace=True, header=19, names=col_names)

        signal_length = len(df.index)
        time = numpy.linspace(start=0, stop=math.ceil(signal_length/fs), num=signal_length)
        sig_dict = {'dataframe': df, 'time': time, 'fs': fs}

        with open('{}/{}.pickle'.format(target_dir, i), 'wb') as handle:
            pickle.dump(sig_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs = '?', type = str, default = "files/", help= "path to data files")
    parser.add_argument('--target', nargs='?', type=str, default='signal_data', help='Target directory for data')
    parser.add_argument('--fs', nargs = '?', type = int, default = 800, help= "sampling frequency")
					
    args = parser.parse_args()
    main(args)
