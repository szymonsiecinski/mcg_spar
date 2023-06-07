import argparse
import pickle
from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC
from tqdm import tqdm
import pandas
from matplotlib.pyplot import figure, plot, savefig, xlabel, ylabel, legend, title, show
import numpy as np
import glob
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import numpy.typing as npt
import sklearn.base as sb


def read_labels(filename):
    with open(filename, 'r') as f:
        return [line.rstrip() for line in f]


def evaluate_classifier(model : sb.BaseEstimator, x: npt.ArrayLike, y: npt.ArrayLike, cv=3, model_name='Base'):
    pred = cross_val_predict(model, x, y, cv=cv)
    cm = confusion_matrix(y, pred)

    precision = precision_score(y, pred, average='macro')
    recall = recall_score(y, pred, average='macro')
    accuracy = accuracy_score(y, pred)

    print(model_name)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))

    accuracies = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
    res_df = pandas.DataFrame(accuracies, index=[0])
    res_df.to_csv('performance_{}.csv'.format(model_name))

    disp=ConfusionMatrixDisplay(cm)
    disp.plot()
    savefig('confmatrix_{}.png'.format(model_name), dpi=150, format='png')


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
    hamming_dists=list()

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
        hamming_dists.append(features['hamming_dist'])

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
                    'v_median': {'Mean': np.mean(v_medians), 'SD': np.std(v_medians), 'Min': np.min(v_medians), 'Max': np.max(v_medians)},
                    'w_median': {'Mean': np.mean(w_medians), 'SD': np.std(w_medians), 'Min': np.min(v_medians), 'Max': np.max(v_medians)},
                    'hamming_dist': {'Mean': np.mean(hamming_dists), 'SD': np.std(hamming_dists), 'Min': np.min(hamming_dists), 'Max': np.max(hamming_dists)},
                    }
        df=pandas.DataFrame(stats_dict)
        df.to_csv('descriptive_stats.csv')

    '''Training a random forest'''
    feature_matrix = np.array([v_mins, v_maxs, v_avgs, v_sds, w_mins, w_maxs, w_avgs, w_sds, hamming_dists]).T
    #feature_matrix = minmax_scale(feature_matrix, feature_range=(-1, 1))
    #x_train, x_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.3, random_state=43)

    '''SMOTE - '''
    smote = SMOTE()
    x, y =smote.fit_resample(feature_matrix, labels)

    tree = RandomForestClassifier(criterion='gini', min_samples_split=2, min_samples_leaf=1)
    svm = SVC(kernel='poly', degree=3, gamma='scale')
    nnetwork = MLPClassifier(hidden_layer_sizes=(6,15,30,30,60,120,60,60,30,30,15,6,), activation='relu', learning_rate='adaptive', solver='sgd')

    evaluate_classifier(tree,feature_matrix, labels, cv=3, model_name='Random Forest (raw)')
    evaluate_classifier(svm,feature_matrix, labels, cv=3, model_name='SVM (raw)')
    evaluate_classifier(nnetwork,feature_matrix, labels, cv=3, model_name='Neural Network (raw)')

    #with SMOTE
    evaluate_classifier(tree, x, y, cv=3, model_name='Random Forest (SMOTE)')
    evaluate_classifier(svm, x, y, cv=3, model_name='SVM (SMOTE)')
    evaluate_classifier(nnetwork, x, y, cv=3, model_name='Neural Network (SMOTE)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs = '?', type = str, default = "spars/", help= "path to data files")
    parser.add_argument('--labels', nargs = '?', type = str, default = "labels.txt", help= "path to labels file")
    parser.add_argument('--signals', nargs='?', type=str, default="ecg", help="type of signals (ecg/scg/gcg)")
    parser.add_argument('--stats', nargs='?', const=True, type=bool, default=False, help="run descriptive statistics")
					
    args = parser.parse_args()
    main(args)
