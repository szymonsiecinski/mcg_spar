import argparse
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, RocCurveDisplay
from tqdm import tqdm
import os
import pandas
from matplotlib.pyplot import figure, plot, savefig, xlabel, ylabel, title, legend, show
import numpy as np
import glob


def save_hr(filename, time, ecg_hr, gcg_hr, fs=800):
    hr = {'time': time, 'ecg_hr': ecg_hr, 'gcg_hr': gcg_hr, 'fs': fs}

    if not(os.path.exists("hr")):
        print("Creating HR folder")
        os.mkdir("hr")

    with open('hr/{}.pickle'.format(filename), 'wb') as f:
        pickle.dump(hr, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_labels(filename):
    with open(filename, 'r') as f:
        return [line.rstrip() for line in f]


def main(args):
    data_path = args.data_path
    signal_type = args.signals
    labels_fname = args.labels

    files = glob.glob('{}/spar_{}_features_*.pickle'.format(data_path, signal_type))
    labels = read_labels(labels_fname)

    v_mins=list()
    v_maxs=list()
    v_avgs=list()
    v_sds=list()
    v_medians=list()
    w_mins=list()
    w_maxs=list()
    w_avgs=list()
    w_sds=list()
    w_medians=list()

    print("Populating lists")
    for i in tqdm(files, total=len(files)):
        # subj_no = [int(s) for s in re.findall(r'\d+', i)]

        with open('{}'.format(i), 'rb') as handle:
            features = pickle.load(handle)

        v_mins.append(features['v_min'])
        v_maxs.append(features['v_max'])
        w_mins.append(features['w_min'])
        w_maxs.append(features['w_max'])
        v_avgs.append(features['v_avg'])
        w_avgs.append(features['w_avg'])
        v_sds.append(features['v_sd'])
        w_sds.append(features['w_sd'])
        v_medians.append(features['v_median'])
        w_medians.append(features['w_median'])

    '''Dump descriptive statistics to csv file'''
    if(args.stats):
        stats_dict={'v_min': {'Mean': np.mean(v_mins), 'SD': np.std(v_mins), 'Min': np.min(v_mins), 'Max': np.max(v_mins)},
                    'v_max': {'Mean': np.mean(v_maxs), 'SD': np.std(v_maxs), 'Min': np.min(v_maxs), 'Max': np.max(v_maxs)},
                    'w_min': {'Mean': np.mean(w_mins), 'SD': np.std(w_mins), 'Min': np.min(w_mins), 'Max': np.max(w_mins)},
                    'w_max': {'Mean': np.mean(w_mins), 'SD': np.std(w_mins), 'Min': np.min(w_mins), 'Max': np.max(w_maxs)},
                    'v_avg': {'Mean': np.mean(v_avgs), 'SD': np.std(v_avgs), 'Min': np.min(v_avgs), 'Max': np.max(v_avgs)},
                    'w_avg': {'Mean': np.mean(v_avgs), 'SD': np.std(v_avgs), 'Min': np.min(v_mins), 'Max': np.max(v_avgs)},
                    'v_sd': {'Mean': np.mean(v_sds), 'SD': np.std(v_sds), 'Min': np.min(v_sds), 'Max': np.max(v_sds)},
                    'w_sd': {'Mean': np.mean(w_sds), 'SD': np.std(w_sds), 'Min': np.min(v_sds), 'Max': np.max(v_sds)},
                    }
        df=pandas.DataFrame(stats_dict)
        df.to_csv('descriptive_stats.csv')

    '''Training a decision tree'''
    if(args.train):
        feature_matrix = np.hstack((np.ndarray(v_mins, dtype=np.float32), np.ndarray(v_maxs, dtype=np.float32)),
                                   np.ndarray(v_avgs, dtype=np.float32), np.ndarray(v_sds, dtype=np.float32),
                                   np.ndarray(v_medians, dtype=np.float32), np.ndarray(w_mins, dtype=np.float32),
                                   np.ndarray(w_maxs, dtype=np.float32), np.ndarray(w_avgs, dtype=np.float32),
                                   np.ndarray(w_sds, dtype=np.float32), np.ndarray(w_medians, dtype=np.float32))
        tree = DecisionTreeClassifier(max_depth=1)
        x_train, x_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.3, random_state=43)
        tree.fit(x_train, y_train)

        acc_train = tree.score(x_train, y_train)
        acc_test = tree.score(x_test, y_test)
        print("Accuracy (training set): {}".format(acc_train))
        print("Accuracy (test set): {}".format(acc_test))

        accuracies = {'acc_train': acc_train, 'acc_test': acc_test}
        res_df = pandas.DataFrame(accuracies)
        res_df.to_csv('accuracies.csv')

        RocCurveDisplay.from_estimator(tree, x_test, y_test, name="ROC", alpha=0.3)
        savefig('ROC.png', dpi=150, format='png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs = '?', type = str, default = "spars/", help= "path to data files")
    parser.add_argument('--labels', nargs = '?', type = str, default = "labels.txt", help= "path to labels file")
    parser.add_argument('--signals', nargs='?', type=str, default="ecg", help="type of signals (ecg/scg/gcg)")
    parser.add_argument('--train', nargs='?', const=True, type=bool, default=False, help="Train a classifier")
    parser.add_argument('--stats', nargs='?', const=True, type=bool, default=False, help="run descriptive statistics")
					
    args = parser.parse_args()
    main(args)