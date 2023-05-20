import numpy as np
from time import time
from collections import defaultdict
from sklearn.linear_model import Ridge, LassoCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ensemble_DV_core_subset import RandomForestClassifierDV_subset
from ensemble_DV_core_original import RandomForestClassifierDV_original
from bagging_DV_core import BaggingClassifierDV, BaggingRegressorDV
from sklearn.isotonic import IsotonicRegression

import warnings
warnings.filterwarnings("ignore")

# def df_oob_agg(df_oob_series):
#     df = df_oob_series.reset_index()
#     df_oob_data = df.groupby('level_0').mean()[0]
#     df_oob_feature = df.groupby('level_1').mean()[0]
#     return df_oob_data, df_oob_feature

class FeatureApproach(object):
    def __init__(self,
                 X, y, 
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

    def run(self,data_oob_run=True, df_oob_run=True, subset_ratio_list=['varying']):
        if data_oob_run:
            self._calculate_proposed_data_oob()
        if df_oob_run:
            self._calculate_proposed_df_oob(subset_ratio_list=subset_ratio_list)
            

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

    def _calculate_proposed_df_oob(self, subset_ratio_list=['varying']):
        print(f'Start: DF-OOB computation')
        # fit a random forest model
        for subset_ratio in subset_ratio_list:
            time_init=time()
            if self.problem == 'clf':
                self.rf_model_subset=RandomForestClassifierDV_subset(n_estimators=self.n_trees, n_jobs=-1) 
            else:
                self.rf_model_subset=RandomForestRegressorDV_subset(n_estimators=self.n_trees, n_jobs=-1) 
            self.rf_model_subset.fit(self.X, self.y, subset_ratio=subset_ratio)
            # self.time_dict['Df-OOB-RF-fitting-%.1f'%subset_ratio]=time()-time_init
            
            time_init=time()
            df_oob_series = self.rf_model_subset.evaluate_dfoob_accuracy(self.X, self.y)
            df_oob = df_oob_series.values.reshape(self.X.shape[0],self.X.shape[1])
            df_oob_data, df_oob_feature = np.mean(df_oob,axis=1), np.mean(df_oob,axis=0)
            
            if subset_ratio == 'varying':
                self.df_value_dict['Df-OOB-%s'%subset_ratio]=df_oob
                self.data_value_dict['Df-OOB-%s'%subset_ratio]=df_oob_data
                self.feature_value_dict['Df-OOB-%s'%subset_ratio]=df_oob_feature
                self.time_dict['Df-OOB-%s'%subset_ratio]=time()-time_init
            else:                
                self.df_value_dict['Df-OOB-%.1f'%subset_ratio]=df_oob
                self.data_value_dict['Df-OOB-%.1f'%subset_ratio]=df_oob_data
                self.feature_value_dict['Df-OOB-%.1f'%subset_ratio]=df_oob_feature
                self.time_dict['Df-OOB-%.1f'%subset_ratio]=time()-time_init

            

        print(f'Done: DF-OOB computation')