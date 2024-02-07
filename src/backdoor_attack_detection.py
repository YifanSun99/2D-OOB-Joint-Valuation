import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_valuation import DataValuation
import numpy as np
import os
import argparse
import pickle

def main(args):
    src_ind, tgt_ind = args.src_ind, args.tgt_ind
    split_no = args.split_no
    attack_type = args.attack_type
    datasets_path = args.datasets_path
    results_path = args.results_path
    class_descr = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dataset = class_descr[src_ind] + '_' + class_descr[tgt_ind]

    data_path = os.path.join(datasets_path,attack_type,dataset,str(split_no))
    save_path = os.path.join(results_path,attack_type,dataset,str(split_no))
    if not os.path.exists(save_path):
        os.makedirs(save_path) 

    X = np.load(os.path.join(data_path, 'train_x.npy'))
    y = np.load(os.path.join(data_path, 'train_y.npy'))
    X_val = np.load(os.path.join(data_path, 'val_x.npy'))
    y_val = np.load(os.path.join(data_path, 'val_y.npy'))
    X_test = np.load(os.path.join(data_path, 'test_x.npy'))
    y_test = np.load(os.path.join(data_path, 'test_y.npy'))
    error_index_train = np.load(os.path.join(data_path, 'error_index_train.npy'))
    error_index_val = np.load(os.path.join(data_path, 'error_index_val.npy'))
    outlier_inds = np.where(error_index_train.flatten() == 1)[0]

    print(X.shape)

    if args.flip_train_label:
        poisoned_training_indices = np.unique(np.where(error_index_train==1)[0])
        y[poisoned_training_indices] = 0
    if args.flip_val_label:
        poisoned_training_indices = np.unique(np.where(error_index_val==1)[0])
        y_val[poisoned_training_indices] = 0

    noisy_index = None
    beta_true = None
    error_row_index = None
    eval_experiments = ['outlier']
    problem = 'clf'
    subset_ratio_list = [0.25,0.5,0.75]
    dargs = {
        'experiment': 'outlier',
        'run_id': 0,
        'n_train': len(X),
        'n_val': len(X_val),
        'n_trees': args.num_trees,
    }

    data_valuation_engine=DataValuation(X=X, y=y, 
        X_val=X_val, y_val=y_val, 
        X_test=X_test, y_test=y_test,
        problem=problem, dargs=dargs)

    data_valuation_engine.compute_feature_shap(subset_ratio_list=subset_ratio_list)
    if not args.remove_baseline:
        data_valuation_engine.prepare_baseline(SHAP_size=None)
    data_valuation_engine.evaluate_data_values(noisy_index, beta_true, error_index_train, error_row_index, X_test, y_test, 
                                                    experiments=eval_experiments,outlier_inds=outlier_inds)


    with open(os.path.join(save_path,'df_values_pattern.pkl'), 'wb')  as f:
        pickle.dump(data_valuation_engine.df_value_dict, f)
    with open(os.path.join(save_path,'outlier_detection_rate.pkl'), 'wb')  as f:
        pickle.dump(data_valuation_engine.outlier_detect_dict, f)

    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ind', type=int, default=6, help='Source index')
    parser.add_argument('--tgt_ind', type=int, default=7, help='Target index')
    parser.add_argument('--split_no', type=int, default=0, help='Split number (default: 0)')
    parser.add_argument('--attack_type', type=str, default='badnets_pattern', help='Type of attack (default: badnets_pattern)')
    parser.add_argument('--datasets_path', type=str, default='./split_datasets/', help='Path to the dataset (default: ./datasets/)')
    parser.add_argument('--results_path', type=str, default='./results/', help='Path to save results (default: ./results/)')
    parser.add_argument('--flip_train_label', action='store_true', default=False, help='Whether flip the labels of poisoned data')
    parser.add_argument('--flip_val_label', action='store_true', default=False, help='Whether flip the labels of poisoned data')
    parser.add_argument('--num_trees', type=int, default=1000, help='Number of trees for random forest')
    parser.add_argument('--remove_baseline', action='store_true', default=False, help='Remove the baselines')

    args = parser.parse_args()
    main(args)