import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, average_precision_score, \
    matthews_corrcoef, cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import prepare_data


class TrainClassifiers:



    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",

    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),

    ]






    def __init__(self, path_to_file):
        self.rng = np.random.RandomState(2)
        p = prepare_data.PrepareData(path_to_file)



        self.X_train, self.X_test, self.y_train, self.y_test = p.get_train_test_split()

        self.X_train_fs, self.X_test_fs = p.select_features(self.X_train, self.X_test, self.y_train)

    def run_classifiers(self, doPrints=True):
        result_list = []
        # iterate over classifiers
        for name, clf in zip(self.names, self.classifiers):
            result_line = {}
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(self.X_train_fs, self.y_train)
            y_pred = clf.predict(self.X_test_fs)


            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            result_line['name'] = name
            result_line['tn'] = tn
            result_line['fp'] = fp
            result_line['fn'] = fn
            result_line['tp'] = tp

            recall = recall_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            matthews_corr = matthews_corrcoef(self.y_test, y_pred)
            cohen_kappa = cohen_kappa_score(self.y_test, y_pred)

            result_line['recall'] = recall

            result_line['precision'] = precision
            result_line['f1'] = f1

            result_line['matthews_corr'] = matthews_corr
            result_line['cohen_kappa'] = cohen_kappa

            result_list.append(result_line)
            if doPrints:
                print('\nfor classifier ' + name)

                print(f'tn {tn}, fp {fp}, fn {fn} , tp {tp}')
                print(f'True Positive Rate : recall {recall}')
                print(f'Positive Predictive Value : precision {precision}')

                print(f'f1 {f1}')
                print(f'matthews_corr {matthews_corr}')
                print(f'cohen_kappa {cohen_kappa}')

        self.result = pd.DataFrame(result_list)
        return self.result





