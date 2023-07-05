import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

import utils

class PrepareData:


    def __init__(self, path_to_file):

        self.df = pd.read_csv(path_to_file)
        #print(self.df.iloc[0])
        #drop columns with uniform variables (y(x) = c for all x)
        self.df = self.df.drop(columns=['EmployeeCount','Over18',  'StandardHours' ])

        #transform categorical columns into numericals
        obj_cols = self.df.select_dtypes(include='object').columns
        self.df[obj_cols] = self.df[obj_cols].apply(lambda col:pd.Categorical(col).codes)

        self.df = self.df.drop(columns=['EmployeeNumber'])
        #print(self.df.iloc[0])
        print('data cleaning done')



    def getXy(self, mask:list =None):
        target = 'Attrition'
        x_cols = list(self.df.columns)

        x_cols.remove(target)
        if mask is not None:
            x_cols = mask



        X = self.df[x_cols].to_numpy()
        y = self.df[target].to_numpy()
        return X,y

    def get_train_test_split(self, mask=None):
        X,y = self.getXy( mask)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_column_names(self):
        return self.df.columns


    def select_features(self,X_train, X_test,y_train, k =15):
        fs = SelectKBest(score_func=f_classif, k=k)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs