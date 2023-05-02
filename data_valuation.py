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

from ensemble_DV_core_subset import RandomForestClassifierDV_subset, RandomForestRegressorDV_subset
from ensemble_DV_core_original import RandomForestClassifierDV_original, RandomForestRegressorDV_original
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from data_approach import DataApproach
from feature_approach import FeatureApproach
import utils_eval

import warnings
warnings.filterwarnings("ignore")

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def rf_score_binary_clf_each_tree(model, X, y):
    B = model.n_estimators
    trees = model.estimators_
    ensemble_features_index = [np.where(i != 0)[0] for i in model._ensemble_features]
    
    accs = [metrics.accuracy_score(y,trees[b].
                                   predict(X[:,ensemble_features_index[b]])) for b in range(B)]
    return np.mean(accs)

def get_values(d,dim):
    values = []
    for i in range(dim):
        values.append(d.get("f%d"%i,0))
    return values

class DataValuation(object):
    def __init__(self, X, y, X_val, y_val, 
                 X_test,y_test,problem, dargs):
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
        self.X_test=X_test
        self.y_test=y_test
        self.problem=problem
        self.dargs=dargs
        self._initialize_instance()        
       
    def _initialize_instance(self):
        # intialize dictionaries
        self.experiment=self.dargs.get('experiment')
        self.run_id=self.dargs.get('run_id')
        self.n_tr=self.dargs.get('n_train')
        self.n_val=self.dargs.get('n_val')
        self.n_trees=self.dargs.get('n_trees')
        self.is_noisy=self.dargs.get('is_noisy')
        self.input_dim=self.dargs.get('input_dim')
        self.rho=self.dargs.get('rho')
        self.base=self.dargs.get('base')
        self.error_row_rate=self.dargs.get('error_row_rate')
        self.error_col_rate=self.dargs.get('error_col_rate')
        self.error_mech=self.dargs.get('error_mech')
        self.mask_ratio=self.dargs.get('mask_ratio')
        self.model_family=self.dargs.get('model_family')
        self.model_name=f'{self.n_tr}:{self.n_val}:{self.n_trees}:{self.is_noisy}'

        # set random seed
        np.random.seed(self.run_id)

        # create placeholders
        self.data_value_dict=defaultdict(list)
        self.feature_value_dict=defaultdict(list)
        self.df_value_dict=defaultdict(list)
        self.time_dict=defaultdict(list)
        self.noisy_detect_dict=defaultdict(list)
        self.mask_detect_dict=defaultdict(list)
        self.rank_dict=defaultdict(list)
        self.feature_removal_dict=defaultdict(list)
        self.error_detect_dict=defaultdict(list)
        self.point_removal_dict=defaultdict(list)
        self.loo_dict=defaultdict(list)


    # def evalute_subset_models(self, X_test, y_test, subset_ratio_list = None):
    #     accs_rf_subset = []
    #     if self.problem == 'clf':
    #         if subset_ratio_list == None:
    #             subset_ratio_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    #         for subset_ratio in subset_ratio_list:
    #             model_rf_subset = RandomForestClassifierDV_subset(n_estimators=1000, n_jobs=-1)
    #             model_rf_subset.fit(self.X, self.y, subset_ratio=subset_ratio)
    #             acc_rf_subset = rf_score_binary_clf_each_tree(model_rf_subset, X_test, y_test)
    #             accs_rf_subset.append(acc_rf_subset)
    #         model_rf_full = RandomForestClassifierDV_subset(n_estimators=1000, n_jobs=-1)
    #         model_rf_full.fit(self.X, self.y, subset_ratio=1.0)
    #         acc_rf_full = rf_score_binary_clf_each_tree(model_rf_full,X_test, y_test)
            
    #         # print(f'RF_subset {acc_rf_subset:.3f}')
    #         print(f'RF_original {acc_rf_full:.3f}')
    #         gaps = []
    #         for acc_rf_subset in accs_rf_subset:
    #             gaps.append(acc_rf_full - acc_rf_subset)            
    #             # print(f'gap_subset {acc_rf_full - acc_rf_subset:.3f}')
    #         self.rf_evaluation_dict={gaps}
    #     else:
    #         raise NotImplementedError('Not implemented yet!')
            
    def compute_data_shap(self, loo_run=True, betashap_run=True):
        '''
        This function computes regular Data-Valuation methods
        '''
        self.data_shap_engine=DataApproach(X=self.X, y=self.y, 
                                            X_val=self.X_val, y_val=self.y_val, 
                                            problem=self.problem, model_family=self.model_family)
        self.data_shap_engine.run(loo_run=False, betashap_run=False)
        self._dict_update(self.data_shap_engine)

    def compute_feature_shap(self, data_oob_run=True, df_oob_run=True, subset_ratio_list=[0.5]):
        '''
        We regard the data valuation problem as feature attribution problem
        This function computes feature attribution methods
        '''
        self.feature_shap_engine=FeatureApproach(X=self.X, y=self.y, 
                                                 problem=self.problem, model_family=self.model_family,
                                                 n_trees=self.n_trees)
        self.feature_shap_engine.run(data_oob_run=data_oob_run,
                                     df_oob_run=df_oob_run,
                                     subset_ratio_list=subset_ratio_list)

        if data_oob_run:
            self.rf_model_original = self.feature_shap_engine.rf_model_original
        self._dict_update(self.feature_shap_engine)

    def _dict_update(self, engine):
        self.data_value_dict.update(engine.data_value_dict)
        self.feature_value_dict.update(engine.feature_value_dict)
        self.df_value_dict.update(engine.df_value_dict)
        self.time_dict.update(engine.time_dict)
        
    def prepare_baseline(self, SHAP_size=1000):
        # base method: treeshap
        print("Start: SHAP computation")
        time_init=time()
        if SHAP_size == None:
            explainer = shap.TreeExplainer(self.rf_model_original)#, sample_X, feature_perturbation = 'interventional')
            shap_values = explainer(self.X)
        else:
            sample_X = self.X[np.random.choice(self.X.shape[0], size=SHAP_size, replace=False), :]
            explainer = shap.TreeExplainer(self.rf_model_original, feature_perturbation = 'interventional')
            shap_values = explainer(sample_X)
        local_importance = np.abs(shap_values.values)[:,:, 0]
        global_importance = local_importance.mean(axis=0)
        self.feature_value_dict['Base'] = global_importance
        self.df_value_dict['Base'] = local_importance
        self.time_dict['Base']=time()-time_init
        
        print("Done: SHAP computation")

    def evaluate_data_values(self, noisy_index, beta_true, error_index, X_test, y_test,
                             experiments, error_row_index):

        
        if 'noisy' in experiments:
            time_init=time()
            self.noisy_detect_dict=utils_eval.noisy_detection_experiment(self.data_value_dict, noisy_index)
            self.time_dict['Eval:noisy']=time()-time_init
            
        if 'point_removal' in experiments:
            time_init=time()
            self.point_removal_dict=utils_eval.point_removal_experiment(self.data_value_dict,
                                                                   self.X, self.y,
                                                                   X_test, y_test,
                                                                   problem=self.problem)
            self.time_dict['Eval:removal']=time()-time_init
            
        if 'mask' in experiments:
            time_init=time()
            mask_index = np.where(beta_true == 0)[0]
            self.mask_detect_dict=utils_eval.mask_detection_experiment(self.feature_value_dict, mask_index)
            self.time_dict['Eval:mask']=time()-time_init
        
        if 'rank' in experiments:
            time_init=time()
            self.rank_dict=utils_eval.rank_experiment(self.feature_value_dict, beta_true)
            self.time_dict['Eval:rank']=time()-time_init

        if 'feature_removal' in experiments:
            time_init=time()
            self.feature_removal_dict=utils_eval.feature_removal_experiment(self.feature_value_dict, self.X, self.y, self.X_test, self.y_test)
            self.time_dict['Eval:feature_removal']=time()-time_init

        # if 'loo' in experiments:
        #     time_init=time()
        #     self.loo_dict=utils_eval.feature_loo_experiment(self.feature_value_dict, self.X, self.y, self.X_test, self.y_test)
        #     self.time_dict['Eval:loo']=time()-time_init

        if 'error' in experiments:
            time_init=time()
            self.error_detect_dict=utils_eval.error_detection_experiment(self.df_value_dict, error_index, error_row_index)
            self.time_dict['Eval:error']=time()-time_init
        
    
    def prepare_learn_oob(self):
        print("Start: Learn-OOB computation")
        time_init=time()
        X_y = np.concatenate((self.X,self.y.reshape(-1,1)), axis=1)
        if 'Data-OOB' not in self.data_value_dict:
            raise ValueError("You should fit Data-OOB first!")
        oob = self.data_value_dict['Data-OOB']
        
        X_y_train, X_y_val, oob_train, oob_val = train_test_split(X_y, oob, test_size=int(0.1 * X_y.shape[0]), random_state=0)
    
        dtrain = xgb.DMatrix(X_y_train, label=oob_train)
        dval = xgb.DMatrix(X_y_val, label=oob_val)
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.01,
            'random_state':0
        }
    
        model = xgb.train(
            params, dtrain, num_boost_round=1000, 
            evals=[(dtrain, 'train'), (dval, 'eval')], 
            early_stopping_rounds=10, 
            verbose_eval=0
        )
        y_pred = model.predict(dval)
        score_mse = mean_squared_error(oob_val, y_pred)
        score_mape = mape(oob_val, y_pred)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_y)
        local_importance = np.abs(shap_values.values)
        global_importance = local_importance.mean(axis=0)
                
        self.feature_value_dict['Learn-OOB'] = global_importance[:-1]
        self.df_value_dict['Learn-OOB'] = local_importance[:,:-1]
        self.time_dict['Learn-OOB']=time()-time_init
        
        
        # weight_importance = np.array(get_values(model.get_score(importance_type='weight'),X_y_train.shape[1]))
        # gain_importance = np.array(get_values(model.get_score(importance_type='gain'),X_y_train.shape[1]))
        
        print("Done: Learn-OOB computation")
        
        return {'X_y_data':(X_y_train,X_y_val),
                'oob_data':(oob_train,oob_val),
                'score_mse':score_mse,
                'score_mape':score_mape,
                'learn_feature_importance':global_importance[:-1],
                # 'weight_importance':weight_importance[:-1],
                # 'gain_importance':gain_importance[:-1],
               }
    

        
    def save_results(self, runpath, dataset, dargs_ind, noisy_index, beta_true):

        print('-'*50)
        print('Save results')
        print('-'*50)
        result_dict={'data_value': self.data_value_dict,
                     'feature_value': self.feature_value_dict,
                     'df_value': self.df_value_dict,
                     'time': self.time_dict,
                     'noisy': self.noisy_detect_dict,
                     'mask': self.mask_detect_dict,
                     "rank":self.rank_dict,
                     'error':self.error_detect_dict,
                     'point_removal': self.point_removal_dict,
                     'feature_removal':self.feature_removal_dict,
                     'loo':self.loo_dict,
                     'dargs':self.dargs,
                     'dataset':dataset,
                     'input_dim':self.X.shape[1],
                     'model_name':self.model_name,
                     'noisy_index': noisy_index,
                     'beta_true': beta_true
                     }
        with open(runpath+f'/run_id{self.run_id}_{dargs_ind}.pkl', 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Done! path: {runpath}, run_id: {self.run_id}.',flush=True)











