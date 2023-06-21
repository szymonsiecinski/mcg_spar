import argparse
import pickle
from sklearn.calibration import cross_val_predict
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import pandas
from matplotlib.pyplot import savefig
import numpy as np
import glob
from imblearn.over_sampling import SMOTE
import numpy.typing as npt
import sklearn.base as sb
from sklearn.utils.multiclass import unique_labels


def read_labels(filename):
    with open(filename, 'r') as f:
        return [line.rstrip() for line in f]


def evaluate_classifier(model : sb.BaseEstimator, x: npt.ArrayLike, y: npt.ArrayLike, cv=3, model_name='Base'):
    model.fit(x, y)
    pred = cross_val_predict(model, x, y, cv=cv)
    labels=unique_labels(y)
    cm = confusion_matrix(y, pred, labels=labels)

    precision = precision_score(y, pred, average='macro', zero_division=0)
    recall = recall_score(y, pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y, pred)

    print(model_name)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))

    accuracies = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
    res_df = pandas.DataFrame(accuracies, index=[0])
    res_df.to_csv('performance_{}.csv'.format(model_name))

    disp=ConfusionMatrixDisplay(cm, display_labels=labels)
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
    vw_dists=list()

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
        vw_dists.append(features['vw_dist'])

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
                    'vw_dists': {'Mean': np.mean(vw_dists), 'SD': np.std(vw_dists), 'Min': np.min(vw_dists), 'Max': np.max(vw_dists)}
                    }
        df=pandas.DataFrame(stats_dict)
        df.to_csv('descriptive_stats.csv')

    '''Training a random forest'''
    feature_matrix = np.array([v_mins, v_maxs, v_avgs, v_sds, v_medians, w_mins, w_maxs, w_avgs, w_sds, w_medians, vw_dists]).T
    rstate = 52

    '''SMOTE - '''
    smote = SMOTE(k_neighbors=3)
    x, y =smote.fit_resample(feature_matrix, labels)

    rforest = RandomForestClassifier(n_estimators=600, criterion='gini', min_samples_split=3, min_samples_leaf=1, n_jobs=-1, random_state=None)
    btrees = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='gini', min_samples_split=3, min_samples_leaf=1),
                               n_estimators=600, n_jobs=-1, random_state=None)
    xgb = GradientBoostingClassifier(n_estimators=600)
    
    svm = SVC(kernel='rbf', degree=3, gamma='scale', random_state=None)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
    loo = LeaveOneOut()

    evaluate_classifier(rforest, feature_matrix, labels, cv=cv, model_name='Random Forest (raw)')
    evaluate_classifier(btrees, feature_matrix, labels, cv=cv, model_name='Bagged Trees (raw)')
    evaluate_classifier(xgb, feature_matrix, labels, cv=cv, model_name='Gradient Boosting (raw)')
    evaluate_classifier(svm,feature_matrix, labels, cv=cv, model_name='SVM (raw)')

    #with SMOTE
    evaluate_classifier(rforest, x, y, cv=cv, model_name='Random Forest (SMOTE)')
    evaluate_classifier(btrees, x, y, cv=cv, model_name='Bagged Trees (SMOTE)')
    evaluate_classifier(xgb, x, y, cv=cv, model_name='Gradient Boosting (SMOTE)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs = '?', type = str, default = "spars/", help= "path to data files")
    parser.add_argument('--labels', nargs = '?', type = str, default = "labels.txt", help= "path to labels file")
    parser.add_argument('--signals', nargs='?', type=str, default="ecg", help="type of signals (ecg/scg/gcg)")
    parser.add_argument('--stats', nargs='?', const=True, type=bool, default=False, help="run descriptive statistics")
					
    args = parser.parse_args()
    main(args)
