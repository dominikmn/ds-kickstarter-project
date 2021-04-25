import json
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from os import mkdir

class kickstarter_predictor():
    
    def __init__(self) -> None:
        self._RSEED=42
        self._json_cols=['category', 'location']

        self._cat_features_impute = ['country', 'currency', 'category_name', 'location_type']

        self._cat_features_onehot = ['country', 'currency', 'category_name', 'location_type']

        self.preprocessor = ColumnTransformer(
            transformers=[
            #('cat_impute', SimpleImputer(strategy='constant', fill_value='missing'), self._cat_features_impute),
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore'), self._cat_features_onehot),
            ('untouched', 'passthrough', ['duration','goal_usd', 'launched_at_month', 'created_at_month'])
            #('untouched', 'passthrough', ['deadline','static_usd_rate', 'goal', 'launched_at', 'created_at'])
            ],
            sparse_threshold=0
        )
        self.model = RandomForestClassifier(n_estimators=120, random_state=self._RSEED, max_features = 'sqrt', n_jobs=-1, verbose = 1)
        try:
            mkdir('./output')
        except OSError:
            print ("Creation of the directory output failed.")
 
    def expand_json_cols(self, df):
        """
        Expand columns that contain json objects

        Parameters
        ---------
        df: Pandas DataFrame

        Returns
        --------
        df: Pandas DataFrame
        """
        df_dicts = pd.DataFrame()
        print('---------- Parsing json ------------')
        for col in self._json_cols:
            print('Parsing json: '+col)
            c = []
            for i, val in df[col].items():
                try:
                    c.append(json.loads(val))
                except:
                    c.append(dict())
            df_dicts[col] = pd.Series(np.array(c))
        print('---------- Expanding dictionaries --------')
        df_expanded = []
        for col in df_dicts.columns:
            print('Expanding: '+col)
            df_expanded.append(pd.json_normalize(df_dicts[col]).add_prefix(col+'_'))
        df = pd.concat([df.drop(self._json_cols, axis=1), pd.concat(df_expanded, axis=1)], axis=1)
        return df

    def data_cleaning(self, df):
        """
        Filter data frame by relevant columns and rows.

        Parameters
        ---------
        df: Pandas DataFrame

        Returns
        --------
        df: Pandas DataFrame
        """
        self.base_features = ['country', 'currency', 'category_name', 'location_type', 'goal', 
                    'launched_at', 'created_at', 'blurb', 'state', 'deadline', 'static_usd_rate']
        df = df[self.base_features]

        #df.dropna(inplace=True)

        df = df.query("state == 'successful' or state == 'failed'")
        dic = {'successful' : 1, 'failed' : 0}
        df['state'] = df['state'].map(dic)

        return df
    
    def feature_engineering(self, df):
        """
        Add custom features

        Parameters
        ---------
        df: Pandas DataFrame

        Returns
        --------
        df: Pandas DataFrame
        """
        df['duration'] = (df.deadline-df.launched_at)/(3600*24)
        df['duration'] = df['duration'].round(2)
        df.drop(['deadline'], axis=1, inplace=True)

        df['goal_usd'] = df['goal'] * df['static_usd_rate']
        df['goal_usd'] = df['goal_usd'].round(2)
        df.drop(['static_usd_rate', 'goal'], axis=1, inplace=True)

        df['launched_at_full'] = pd.to_datetime(df['launched_at'], unit='s')
        df['launched_at_month'] = pd.DatetimeIndex(df['launched_at_full']).month
        df.drop(['launched_at', 'launched_at_full'], axis=1, inplace=True)

        df['created_at_full'] = pd.to_datetime(df['created_at'], unit='s')
        df['created_at_month'] = pd.DatetimeIndex(df['created_at_full']).month
        df.drop(['created_at', 'created_at_full'], axis=1, inplace=True)

        df['blurb_len'] = [(x.split(" ") if isinstance(x, str) else "") for x in df.blurb]
        df['blurb_len'] = [len(i) for i in df['blurb_len']]
        df.drop(['blurb'], axis=1, inplace=True)
        return df

    def read_csv(self, name):
        """
        Read csv file in kickstarter format

        Parameters
        ---------
        name: String. Only for display purposes.

        Returns
        --------
        df: Pandas DataFrame
        """
        file_name = input(f"Please enter {name} csv file name: ")
        if(not file_name): 
            file_name = './data/Kickstarter003.csv'
            print(f'Taking default file {file_name}')
        return pd.read_csv(file_name)
    
    def processor_lossy(self, df):
        """
        Apply data frame preprocessing. Outside of sklearn.pipeline

        Parameters
        ---------
        df: Pandas DataFrame

        Returns
        --------
        df: Pandas DataFrame
        """
        df = self.expand_json_cols(df)
        df = self.data_cleaning(df)
        X = df.drop('state', axis=1)
        y = df.state
        return X, y
   
    def dump_model(self): 
        #r = f"{rmse_score_final:.0f}".replace('.','') 
        t = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        o = f"./output/model_dump_{t}.pickle"
        print(f'Dumping model to pickle: {o}')
        dump(self.model, o)

    def model_fit_and_export(self):
        """
        Wrapper for fit and export tasks

        Parameters
        ---------
        None

        Returns
        --------
        None
        """
        df = self.read_csv('train')
        self.X_train, self.y_train = self.processor_lossy(df)
        self.X_train = self.feature_engineering(self.X_train)
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.model.fit(self.X_train, self.y_train)
        self.dump_model()

    def model_load(self):
        """
        Load model from pickle file and store in class attribute.

        Parameters
        ---------
        None

        Returns
        --------
        None
        """
        model_file_name = ''
        model_file_name = input('Please enter model file name: ')
        if(model_file_name):
            self.model=load(model_file_name)
        else:
            print('Taking previously trained model.')

    def printscore(self): 
        print(classification_report(self.y_test, self.y_pred)) 
    
    def prediction_tocsv(self):
        #r = f"{rmse_score_final:.0f}".replace('.','') 
        t = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        o = f"./output/y_pred_{t}.csv"
        print(f'Writing prediction to csv: {o}')
        pd.DataFrame(self.y_pred).to_csv(o, index = False)
    
    def readcsv_and_predict(self):
        """
        Wrapper for read and predict tasks

        Parameters
        ---------
        None

        Returns
        --------
        None
        """
        df = self.read_csv('test')
        self.X_test, self.y_test = self.processor_lossy(df)  
        self.X_test = self.feature_engineering(self.X_test)
        self.X_test = self.preprocessor.transform(self.X_test)
        self.y_pred = self.model.predict(self.X_test)
        self.printscore()
        self.prediction_tocsv()

def main():
    ks = kickstarter_predictor()

    ks.model_fit_and_export()

    ks.model_load()
    ks.readcsv_and_predict()
    
if __name__ == '__main__':
    main()
