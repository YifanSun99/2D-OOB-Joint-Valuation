from time import time
import numpy as np
import pickle
import shap
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import metrics

from ensemble_DV_core import RandomForestClassifierDV, RandomForestRegressorDV
from ensemble_DV_core_original import RandomForestClassifierDV_original, RandomForestRegressorDV_original

from data_approach import DataApproach
from feature_approach import FeatureApproach
import utils_eval

import warnings
warnings.filterwarnings("ignore")

class DataValuation(object):
    def __init__(self, X, y, X_val, y_val, problem, dargs):
        """
        Args:
            (X, y): (inputs, outputs) to be valued.
            (X_val, y_val): (inputs, outputs) to be used for utility evaluation.
            problem: "clf" 
            dargs: arguments of the experimental setting
        """
        self.X=X
        self.y=y
        self.X_val=X_val
        self.y_val=y_val
        self.problem=problem
        self.dargs=dargs
        self._initialize_instance()        
       
    def _initialize_instance(self):
        # intialize dictionaries
        self.run_id=self.dargs['run_id']
        self.n_tr=self.dargs['n_data_to_be_valued']
        self.n_val=self.dargs['n_val']
        self.n_trees=self.dargs['n_trees']
        self.is_noisy=self.dargs['is_noisy']
        self.model_family=self.dargs['model_family']
        self.model_name=f'{self.n_tr}:{self.n_val}:{self.n_trees}:{self.is_noisy}'

        # set random seed
        np.random.seed(self.run_id)

        # create placeholders
        self.data_value_dict=defaultdict(list)
        self.feature_value_dict=defaultdict(list)
        self.df_value_dict=defaultdict(list)
        self.time_dict=defaultdict(list)
        self.noisy_detect_dict=defaultdict(list)
        self.masked_detect_dict=defaultdict(list)
        self.removal_dict=defaultdict(list)
        self.learn_dict=defaultdict(list)
        self.rank_dict=defaultdict(list)
        self.new_evaluation_dict=defaultdict(list)

    def evaluate_baseline_models(self, X_test, y_test):
        if self.problem == 'clf':
            model_logistic=LogisticRegression()
            model_logistic.fit(self.X, self.y)
            acc_logistic=model_logistic.score(X_test, y_test)

            model_rf=RandomForestClassifier()
            model_rf.fit(self.X, self.y)
            acc_rf=model_rf.score(X_test, y_test)
            acc_tree_1=np.mean([model_tmp.score(X_test, y_test) for model_tmp in model_rf.estimators_])

            model_tree=DecisionTreeClassifier()
            model_tree.fit(self.X, self.y)
            acc_tree=model_tree.score(X_test, y_test)

            model_knn=KNeighborsClassifier(n_neighbors=10)
            model_knn.fit(self.X, self.y)
            acc_knn=model_knn.score(X_test, y_test)

            print(f'Logistic {acc_logistic:.3f}')
            print(f'RF {acc_rf:.3f}')
            print(f'Avg tree {acc_tree_1:.3f}')
            print(f'Tree {acc_tree:.3f}')
            print(f'KNN {acc_knn:.3f}')
            self.baseline_score_dict={'Meta_Data': ['Logistic', 'RF', 'Avg_tree', 'Tree', 'KNN'],
                                      'Results': [acc_logistic, acc_rf, acc_tree_1, acc_tree, acc_knn]}
        else:
            model_linear=LinearRegression()
            model_linear.fit(self.X, self.y)
            r2_linear=model_linear.score(X_test, y_test)

            model_rf=RandomForestRegressor()
            model_rf.fit(self.X, self.y)
            r2_rf=model_rf.score(X_test, y_test)
            r2_tree_1=np.mean([model_tmp.score(X_test, y_test) for model_tmp in model_rf.estimators_])

            model_tree=DecisionTreeRegressor()
            model_tree.fit(self.X, self.y)
            r2_tree=model_tree.score(X_test, y_test)

            model_knn=KNeighborsRegressor(n_neighbors=10)
            model_knn.fit(self.X, self.y)
            r2_knn=model_knn.score(X_test, y_test)

            print(f'Linear {r2_linear:.3f}')
            print(f'RF {r2_rf:.3f}')
            print(f'Avg tree {r2_tree_1:.3f}')
            print(f'Tree {r2_tree:.3f}')
            print(f'KNN {r2_knn:.3f}')
            self.baseline_score_dict={'Meta_Data': ['Linear', 'RF', 'Avg_tree', 'Tree', 'KNN'],
                                      'Results': [r2_linear, r2_rf, r2_tree_1, r2_tree, r2_knn]}
            
    
    
    def evalute_rf_models(self, X_test, y_test):
        if self.problem == 'clf':
            model_rf = RandomForestClassifierDV(n_estimators=800, n_jobs=-1)
            model_rf.fit(self.X, self.y)
            acc_rf = model_rf.score(X_test, y_test)

            model_rf_original = RandomForestClassifierDV_original(n_estimators=800, n_jobs=-1)
            model_rf_original.fit(self.X, self.y)
            acc_rf_original = model_rf_original.score(X_test, y_test)
            
            print(f'RF {acc_rf:.3f}')
            print(f'RF_original {acc_rf_original:.3f}')
            print(f'gap {acc_rf_original - acc_rf:.3f}')
            
            self.rf_evaluation_dict={'Meta_Data': [ 'RF', 'RF_original','Gap'],
                                      'Results': [acc_rf, acc_rf_original,acc_rf_original - acc_rf]}
        else:
            raise NotImplementedError('Not implemented yet!')
            
    def compute_data_shap(self, loo_run=True, betashap_run=True):
        '''
        This function computes regular Data-Valuation methods
        '''
        self.data_shap_engine=DataApproach(X=self.X, y=self.y, 
                                           X_val=self.X_val, y_val=self.y_val, 
                                           problem=self.problem, model_family=self.model_family)
        self.data_shap_engine.run(loo_run=loo_run, betashap_run=betashap_run)
        self._dict_update(self.data_shap_engine)

    def compute_feature_shap(self, AME_run=True, lasso_run=True, boosting_run=True, treeshap_run=True, simple_run=False, data_oob_run=True, df_oob_run=True):
        '''
        We regard the data valuation problem as feature attribution problem
        This function computes feature attribution methods
        '''
        self.feature_shap_engine=FeatureApproach(X=self.X, y=self.y, 
                                                 X_val=self.X_val, y_val=self.y_val, 
                                                 problem=self.problem, model_family=self.model_family,
                                                 n_trees=self.n_trees)
        self.feature_shap_engine.run(AME_run=AME_run,
                                     lasso_run=lasso_run, 
                                     boosting_run=boosting_run,
                                     treeshap_run=treeshap_run,
                                     simple_run=simple_run,
                                     data_oob_run=data_oob_run,
                                     df_oob_run=df_oob_run)
        if df_oob_run:
            self.rf_model = self.feature_shap_engine.rf_model
        if data_oob_run:
            self.rf_model_original = self.feature_shap_engine.rf_model_original
        self._dict_update(self.feature_shap_engine)

    def _dict_update(self, engine):
        self.data_value_dict.update(engine.data_value_dict)
        self.feature_value_dict.update(engine.feature_value_dict)
        self.df_value_dict.update(engine.df_value_dict)
        self.time_dict.update(engine.time_dict)

    def evaluate_data_values(self, noisy_index, beta_true, X_test, y_test, removal_run=True, SHAP_run=True):
        if SHAP_run:
            # base method: treeshap
            print("Start: SHAP computation")
            sample_X = self.X[np.random.choice(self.X.shape[0], size=1000, replace=False), :]
            time_init=time()
            explainer = shap.TreeExplainer(self.rf_model, sample_X, feature_perturbation = 'interventional')
            shap_values = explainer(sample_X)
            global_importance = np.abs(shap_values.values).mean(axis=0)[:,0]
            self.feature_value_dict['Base'] = global_importance
            self.time_dict['Base']=time()-time_init
            
            # time_init=time()
            # explainer_original = shap.TreeExplainer(self.rf_model_original, sample_X, feature_perturbation = 'interventional')
            # shap_values_original = explainer_original(sample_X)
            # global_importance_original = np.abs(shap_values_original.values).mean(axis=0)[:,0]
            # self.feature_value_dict['Base-original'] = global_importance_original
            # self.time_dict['Base-original']=time()-time_init
            print("Done: SHAP computation")

        
        if self.dargs['is_noisy'] > 0:
            time_init=time()
            self.noisy_detect_dict=utils_eval.noisy_detection_experiment(self.data_value_dict, noisy_index)
            self.time_dict['Eval:noisy']=time()-time_init
        
        time_init=time()
        masked_index = np.where(beta_true == 0)[0]
        self.masked_detect_dict=utils_eval.masked_detection_experiment(self.feature_value_dict, masked_index)
        self.time_dict['Eval:masked']=time()-time_init
        
        if removal_run is True:
            time_init=time()
            self.removal_dict=utils_eval.point_removal_experiment(self.data_value_dict,
                                                                   self.X, self.y,
                                                                   X_test, y_test,
                                                                   problem=self.problem)
            self.time_dict['Eval:removal']=time()-time_init
        
        time_init=time()
        self.rank_dict=utils_eval.rank_experiment(self.feature_value_dict, beta_true)
        self.time_dict['Eval:rank']=time()-time_init
        
        
        
        
    def save_results(self, runpath, dataset, dargs_ind, noisy_index, beta_true):
        self.sparsity_dict=defaultdict(list)
        for key in self.data_value_dict:
            self.sparsity_dict[key]=np.mean(self.data_value_dict[key]==0)

        print('-'*50)
        print('Save results')
        print('-'*50)
        result_dict={'data_value': self.data_value_dict,
                     'feature_value': self.feature_value_dict,
                     'df_value': self.df_value_dict,
                     'sparse': self.sparsity_dict,
                     'time': self.time_dict,
                     'noisy': self.noisy_detect_dict,
                     'mask': self.masked_detect_dict,
                     'removal': self.removal_dict,
                     'dargs':self.dargs,
                     'dataset':dataset,
                     'input_dim':self.X.shape[1],
                     'model_name':self.model_name,
                     'noisy_index': noisy_index,
                     'beta_true': beta_true,
                     'learn': self.learn_dict,
                     "rank":self.rank_dict,
                     # 'baseline_score':self.baseline_score_dict,
                     # 'rf_score':self.rf_evaluation_dict,
                     'new':self.new_evaluation_dict}
                     
        with open(runpath+f'/run_id{self.run_id}_{dargs_ind}.pkl', 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Done! path: {runpath}, run_id: {self.run_id}.',flush=True)











