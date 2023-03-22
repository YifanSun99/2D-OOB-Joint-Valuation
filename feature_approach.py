import numpy as np
from time import time
from collections import defaultdict
from sklearn.linear_model import Ridge, LassoCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ensemble_DV_core import RandomForestClassifierDV, RandomForestRegressorDV
from ensemble_DV_core_original import RandomForestClassifierDV_original, RandomForestRegressorDV_original
from bagging_DV_core import BaggingClassifierDV, BaggingRegressorDV
from sklearn.isotonic import IsotonicRegression

import warnings
warnings.filterwarnings("ignore")

def df_oob_agg(df_oob_series):
    df = df_oob_series.reset_index()
    df_oob_data = df.groupby('level_0').mean()[0]
    df_oob_feature = df.groupby('level_1').mean()[0]
    return df_oob_data, df_oob_feature

class FeatureApproach(object):
    def __init__(self,
                 X, y, 
                 X_val, y_val, 
                 problem, model_family,
                 n_trees):
        """
        Args:
            (X,y): (inputs,outputs) to be valued.
            (X_val,y_val): (inputs,outputs) to be used for utility evaluation.
            problem: "clf"
            model_family: The model family used for learning algorithm
            GR_threshold: Gelman-Rubin threshold for convergence criteria
            max_iters: maximum number of iterations (for a fixed cardinality)
        """
        self.X=X
        self.y=y
        self.X_val=X_val
        self.y_val=y_val
        self.problem=problem
        self.model_family=model_family
        self.n_trees=n_trees
        self._initialize_instance()   

    def _initialize_instance(self):
        # create placeholders
        self.data_value_dict=defaultdict(list)
        self.feature_value_dict=defaultdict(list)
        self.df_value_dict=defaultdict(list)
        self.time_dict=defaultdict(list) 

    def run(self, AME_run=True, lasso_run=True, boosting_run=True, treeshap_run=True, bagging_run=True, simple_run=False, data_oob_run=True, df_oob_run=True):
        if data_oob_run:
            self._calculate_proposed_data_oob()
        if df_oob_run:
            self._calculate_proposed_df_oob()
        if AME_run is True: self._calculate_AME()
            
    def _calculate_AME(self):
        print(f'Start: AME computation')
        # fit AME model
        time_init=time()
        X_dv_ame_list, y_dv_ame_list=[],[]
        N_to_be_valued=len(self.y)
        if self.problem == 'clf':
            for max_sample in [0.2, 0.4, 0.6, 0.8]:
                AME_clf=BaggingClassifierDV(n_estimators=(self.n_trees//4), 
                                            estimator=DecisionTreeClassifier(), 
                                            max_samples=max_sample,
                                            bootstrap=False,
                                            n_jobs=-1) 
                AME_clf.fit(self.X, self.y)

                # create the data_valuation dataset
                X_dv_ame, y_dv_ame=AME_clf.evaluate_importance(self.X_val, self.y_val)
                X_dv_ame_list.append(X_dv_ame)
                y_dv_ame_list.append(y_dv_ame)
        else:
            for max_sample in [0.2, 0.4, 0.6, 0.8]:
                AME_model=BaggingRegressorDV(n_estimators=(self.n_trees//4), 
                                            estimator=DecisionTreeRegressor(), 
                                            max_samples=max_sample,
                                            bootstrap=False,
                                            n_jobs=-1) 
                AME_model.fit(self.X, self.y)

                # create the data_valuation dataset
                X_dv_ame, y_dv_ame=AME_model.evaluate_importance(self.X_val, self.y_val)
                X_dv_ame_list.append(X_dv_ame)
                y_dv_ame_list.append(y_dv_ame)
        
        X_dv_ame_list=np.vstack(X_dv_ame_list)
        y_dv_ame_list=np.vstack(y_dv_ame_list).reshape(-1)

        # normalize X and y
        X_dv_ame_list=((X_dv_ame_list.T-np.mean(X_dv_ame_list, axis=1))/(np.mean(X_dv_ame_list, axis=1)*(1-np.mean(X_dv_ame_list, axis=1)))).T 
        y_dv_ame_list=y_dv_ame_list-np.mean(y_dv_ame_list)

        dv_ame=LassoCV()
        dv_ame.fit(X=X_dv_ame_list, y=y_dv_ame_list)
        self.data_value_dict['AME']=dv_ame.coef_
        self.time_dict['AME']=time()-time_init
        print(f'Done: AME computation')

    def _calculate_proposed_data_oob(self):
        print(f'Start: Data-OOB computation')
        # fit a random forest model
        time_init=time()
        if self.problem == 'clf':
            self.rf_model_original=RandomForestClassifierDV_original(n_estimators=self.n_trees, n_jobs=-1) 
        else:
            self.rf_model_original=RandomForestRegressorDV_original(n_estimators=self.n_trees, n_jobs=-1) 
        self.rf_model_original.fit(self.X, self.y)
        self.time_dict['Data_oob_RF_fitting']=time()-time_init
        self.data_value_dict['Data-OOB']=(self.rf_model_original.evaluate_oob_accuracy(self.X, self.y)).to_numpy()
        self.time_dict['Data-OOB']=time()-time_init
        print(f'Done: Data-OOB computation')

    def _calculate_proposed_df_oob(self):
        print(f'Start: DF-OOB computation')
        # fit a random forest model
        time_init=time()
        if self.problem == 'clf':
            self.rf_model=RandomForestClassifierDV(n_estimators=self.n_trees, n_jobs=-1) 
        else:
            self.rf_model=RandomForestRegressorDV(n_estimators=self.n_trees, n_jobs=-1) 
        self.rf_model.fit(self.X, self.y)
        self.time_dict['DF_oob_RF_fitting']=time()-time_init
        df_oob_series = self.rf_model.evaluate_dfoob_accuracy(self.X, self.y)
        self.df_value_dict['Df-OOB'] = df_oob_series
        
        df_oob_data, df_oob_feature = df_oob_agg(df_oob_series)
        self.data_value_dict['Df-OOB-data']=df_oob_data.to_numpy()
        self.feature_value_dict['Df-OOB-feature']=df_oob_feature.to_numpy()
        self.feature_value_dict['Df-OOB-error-feature']=1-df_oob_feature.to_numpy()
        self.time_dict['Df-OOB']=time()-time_init
        print(f'Done: DF-OOB computation')