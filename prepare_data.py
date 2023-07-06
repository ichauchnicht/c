'''
A little object doing the first data preparation steps: removing uniform and index-like columns,
transforming categorical data to numerical.
'''
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

import utils


class PrepareData:

    def __init__(self, path_to_file):
        self.df = pd.read_csv(path_to_file)

    def prepare_data(self):
        '''
        basic data prepration steps.
        '''
        # drop columns with uniform variables (y(x) = c for all x)
        # todo column names should not be hardcoded
        self.df = self.df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'])

        # transform categorical columns into numericals
        obj_cols = self.df.select_dtypes(include='object').columns
        self.df[obj_cols] = self.df[obj_cols].apply(lambda col: pd.Categorical(col).codes)

        # todo column names should not be hardcoded
        self.df = self.df.drop(columns=['EmployeeNumber'])
        print('data cleaning done')

    def getXy(self, target='Attrition', mask: list = None):
        '''
        returns the data as numpy arrays suitable for ML algorithms. The target column is
        :param target: target column (y) default = 'Attrition'
        :param mask: a list of columns to be used only as feature columns
        :return: X and y as data
        '''

        x_cols = list(self.df.columns)

        x_cols.remove(target)
        if mask is not None:
            x_cols = mask

        X = self.df[x_cols].to_numpy()
        y = self.df[target].to_numpy()
        return X, y

    def get_train_test_split(self, target='Attrition', mask=None):
        '''
        use sklearn's method to split data set into train and test subset.
        :param target: target column (y) default = 'Attrition'
        :param mask: a list of columns to be used only as feature columns
        :return: train and test subsets
        '''
        X, y = self.getXy(target, mask)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_column_names(self):
        return self.df.columns

    def select_features(self, X_train, X_test, y_train, k=15):
        ''' use sklearn's f_classif to run feature selection. return top k feaatures, default k=15'''
        fs = SelectKBest(score_func=f_classif, k=k)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs
