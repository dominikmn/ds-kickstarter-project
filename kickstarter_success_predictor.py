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
 
    def expand_json_cols(self, df):
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
        print('---------- Data Cleaning --------')
        print(f"Shape before clean: {df.shape}")
        self.base_features = ['country', 'currency', 'category_name', 'location_type', 'goal', 
                    'launched_at', 'created_at', 'blurb', 'state', 'deadline', 'static_usd_rate']
        df = df[self.base_features]

        #df.dropna(inplace=True)

        df = df.query("state == 'successful' or state == 'failed'")
        dic = {'successful' : 1, 'failed' : 0}
        df.loc[:,'state'] = df.state.map(dic)
        print(f"Shape after clean: {df.shape}")

        return df
    
    def feature_engineering(self, df):
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
        file_name = input(f"Please enter {name} csv file name: ")
        if(not file_name): 
            file_name = './data/Kickstarter003.csv'
            print(f'Taking default file {file_name}')
        return pd.read_csv(file_name)
    
    def processor_lossy(self, df):
        df = self.expand_json_cols(df)
        df = self.data_cleaning(df)
        X = df.drop('state', axis=1)
        y = df.state
        return X, y
   
    def dump_model(self): 
        #r = f"{rmse_score_final:.0f}".replace('.','') 
        t = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dump(self.model, f"model_dump_{t}.pickle")

    def model_fit_and_export(self):
        df = self.read_csv('train')
        self.X_train, self.y_train = self.processor_lossy(df)
        #df_imputed = self.imputer.fit_transform(self.X_train[self._cat_features_impute])
        #df_onehot = self.onehotenc.fit_transform(self.X_train[self._cat_features_onehot])
        self.X_train = self.feature_engineering(self.X_train)
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        print(self.X_train.shape)
        self.model.fit(self.X_train, self.y_train)
        self.dump_model()

    def model_load(self):
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
        pd.DataFrame(self.y_pred).to_csv(f"y_pred_{t}.csv", index = False)
    
    def readcsv_and_predict(self):
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
