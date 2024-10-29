# Imports
import numpy as np
import pandas as pd
import tqdm
from time import time
import os

import configs
import datasets
from ensemble_DV_core_subset import RandomForestClassifierDV_subset
from ensemble_DV_core_original import RandomForestClassifierDV_original
from data_valuation import DataValuation
import utils_eval
import matplotlib.pyplot as plt


def run_experiment_core(config):
    print(config)
    runpath = config['runpath']
    if not os.path.exists(runpath):
        os.makedirs(runpath)

    problem = config['problem']
    dataset = config['dataset']
    dargs_list = config['dargs_list']
    experiment = dargs_list[0]['experiment']
    print(experiment)

    if experiment == 'normal':
        eval_experiments = ['point_removal','cell_removal']#,'feature_removal']
    elif experiment == 'noisy':
        eval_experiments = ['point_removal','noisy']
    elif experiment == 'error':
        raise NotImplementedError
        # eval_experiments = ['error', 'point_removal']
    elif experiment == 'outlier':
        eval_experiments = ['outlier', 'cell_removal']

    for dargs_ind in range(len(dargs_list)):
        dargs = dargs_list[dargs_ind]
        (X, y), (X_val, y_val), (X_test, y_test), \
                                noisy_index, beta_true, error_index, \
                                error_row_index, X_original = \
                                datasets.load_data(problem,dataset,**dargs)
        data_valuation_engine=DataValuation(X=X, y=y, 
                                            X_val=X_val, y_val=y_val, 
                                            X_test=X_test, y_test=y_test,
                                            problem=problem, dargs=dargs)
        
        if experiment == 'outlier':
            outlier_inds = np.where(error_index.flatten() == 1)[0]
        else:
            outlier_inds = None
        #     outlier_inds_1 = []
        #     for i in outlier_inds:
        #         row = i // X.shape[1]
        #         cur_label = y[row]
        #         if cur_label == 0:
        #             outlier_inds_0.append(i)
        #         elif cur_label == 1:
        #             outlier_inds_1.append(i)
        # else:
        #     outlier_inds_0=None
        #     outlier_inds_1=None
        if experiment in ['noisy','normal']:
            data_valuation_engine.compute_data_shap()
            data_valuation_engine.prepare_data_valuation_baseline()
        data_valuation_engine.compute_feature_shap(subset_ratio_list=[0.25,0.75])
        if experiment in ['noisy','normal','error','outlier']:
            data_valuation_engine.prepare_baseline(SHAP_size=None)
        data_valuation_engine.evaluate_data_values(noisy_index, beta_true, error_index, error_row_index, X_test, y_test, 
                                                   experiments=eval_experiments,outlier_inds=outlier_inds)
                                                            # experiments=eval_experiments,outlier_inds_0=outlier_inds_0,outlier_inds_1=outlier_inds_1)
        data_valuation_engine.save_results(runpath, dataset, dargs_ind, noisy_index, beta_true,
                                            error_index=error_index,error_row_index=error_row_index)
        del data_valuation_engine
    