import numpy as np
import pandas as pd
import pickle
from scipy.stats import t
import tqdm
from scipy.stats import norm

def extreme_prob(x, mean, std):
    z = (x - mean) / std
    return norm.cdf(-np.abs(z)) + (1 - norm.cdf(np.abs(z)))

def add_outliers(X, y, outlier_addition = 'two_stage'):
    def normpdf(x, mean, std):
        var = float(std)**2
        denom = np.sqrt(2*np.pi*var)
        num = np.exp(-(float(x)-float(mean))**2/(2*var))
        return num / denom
    class_0_inds = np.where(y == 0)[0]
    class_1_inds = np.where(y == 1)[0]
    
    X_class_0 = X[class_0_inds]
    X_class_1 = X[class_1_inds]
    
    feat_means_0 = []
    feat_stds_0 = []
    for i in range(X.shape[1]):
        feat_i = X_class_0[:, i]
        feat_means_0.append(np.mean(feat_i))
        feat_stds_0.append(np.std(feat_i))
    
    feat_means_1 = []
    feat_stds_1 = []
    for i in range(X.shape[1]):
        feat_i = X_class_1[:, i]
        feat_means_1.append(np.mean(feat_i))
        feat_stds_1.append(np.std(feat_i))
    
    all_feat_stats = {0: [feat_means_0, feat_stds_0, 0.01], 1: [feat_means_1, feat_stds_1, 0.01]}

    selected_rows = np.random.choice(X.shape[0], size=int(0.4*X.shape[0]), replace=False)
    outlier_inds_list = []
    for row in selected_rows:
        selected_cols = np.random.choice(X.shape[1], size=max(int(0.4*X.shape[1]),1), replace=False)
        for col in selected_cols:
            flattened_index = row * X.shape[1] + col
            outlier_inds_list.append(flattened_index)
    outlier_inds = np.sort(np.array(outlier_inds_list))

    X_with_outliers = X.copy()
    outlier_mask = np.zeros_like(X)
    outlier_row_index = []
    
    for i in outlier_inds:
        row = i // X.shape[1]
        outlier_row_index.append(row)
        col = i % X.shape[1]
        cur_label = y[row]
        
        feat_mean, feat_std, likeli = all_feat_stats[cur_label]
        norm_val = 1
        feat = 0

        while norm_val > likeli:
            feat = np.random.normal(0,1)
            norm_val = extreme_prob(feat, feat_mean[col], feat_std[col])

        X_with_outliers[row, col] = feat
        outlier_mask[row, col] = 1
    
    return X_with_outliers, outlier_mask, np.array(list(set(outlier_row_index)))

def load_data(problem, dataset, **dargs):
    '''
    (X,y): data to be valued
    (X_val, y_val): data to be used for evaluation
    (X_test, y_test): data to be used for downstream ML tasks
    '''
    print('-'*30)
    print(dargs)
        
    if problem=='reg':
        raise NotImplementedError('Reg problem not implemented yet!')
    elif problem=='clf':
        (X, y), (X_val, y_val), (X_test, y_test), beta_true, error_index, error_row_index, X_original = load_classification_dataset(dataset=dataset,
                                                                            experiment=dargs.get('experiment'),
                                                                            n_train=dargs['n_train'],
                                                                            n_val=dargs['n_val'],
                                                                            n_test=dargs['n_test'],
                                                                            input_dim=dargs.get('input_dim'),
                                                                            clf_path=dargs.get('clf_path'),
                                                                            openml_path=dargs.get('openml_clf_path'),
                                                                            mask_ratio=dargs.get('mask_ratio'),
                                                                            rho=dargs.get('rho'),
                                                                            base=dargs.get('base'),
                                                                            error_row_rate=dargs.get('error_row_rate'),
                                                                            error_col_rate=dargs.get('error_col_rate'),
                                                                            error_mech=dargs.get('error_mech'))
        if dargs['experiment'] == 'noisy':
            n_class=len(np.unique(y))

            # training is flipped
            noisy_index=np.random.choice(np.arange(dargs['n_train']), 
                                           int(dargs['n_train']*dargs['is_noisy']), 
                                           replace=False) 
            random_shift=np.random.choice(n_class-1, len(noisy_index), replace=True)
            y[noisy_index]=(y[noisy_index] + 1 + random_shift) % n_class

            # validation is also flipped
            noisy_val_index=np.random.choice(np.arange(dargs['n_val']),
                                               int(dargs['n_val']*dargs['is_noisy']), 
                                               replace=False) 
            random_shift=np.random.choice(n_class-1, len(noisy_val_index), replace=True)
            y_val[noisy_val_index]=(y_val[noisy_val_index] + 1 + random_shift) % n_class 
        else:
            noisy_index = None

        return (X, y), (X_val, y_val), (X_test, y_test), noisy_index, beta_true, error_index, error_row_index, X_original
    else:
        raise NotImplementedError('Check problem')


def load_classification_dataset(dataset,
                                experiment,
                                n_train, 
                                n_val, 
                                n_test, 
                                input_dim,
                                clf_path='clf_path',
                                openml_path='openml_path',
                                mask_ratio=0.5,
                                rho=0,
                                base=3,
                                error_row_rate=0.1,
                                error_col_rate=0.1,
                                error_mech='noise'
                                ):
    '''
    This function loads classification datasets.
    n_train: The number of data points to be valued.
    n_val: Validation size. Validation dataset is used to evalute utility function.
    n_test: Test size. Test dataset is used to evalute model performance.
    clf_path: path to classification datasets.
    openml_path: path to openml datasets.
    '''
    if dataset == 'gaussian':
        print('-'*50)
        print('GAUSSIAN-C')
        print('-'*50)
        n, input_dim=max(100000, n_train+n_val+n_test+1), input_dim
        
        if rho != 0:
            U_cov = np.diag((1-rho)*np.ones(input_dim))+rho
            U_mean = np.zeros(input_dim)
            data = np.random.multivariate_normal(U_mean, U_cov, n)
        else:
            data = np.random.normal(size=(n,input_dim))

        if experiment == 'noisy' or experiment == 'normal' or experiment == 'outlier':
            # beta
            beta_true = np.random.normal(size=(input_dim,1))
        elif experiment == 'error':
            # beta
            beta_true = np.random.normal(loc = 1,scale = np.sqrt(0.1), size=(input_dim,1))
        p_true = np.exp(data.dot(beta_true))/(1.+np.exp(data.dot(beta_true)))
        target = np.random.binomial(n=1, p=p_true).reshape(-1)
    elif dataset == 'pol':
        print('-'*50)
        print('pol')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/pol_722.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'jannis':
        print('-'*50)
        print('jannis')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/jannis_43977.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'lawschool':
        print('-'*50)
        print('law-school-admission-bianry')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/law-school-admission-bianry_43890.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'fried':
        print('-'*50)
        print('fried')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/fried_901.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'vehicle_sensIT':
        print('-'*50)
        print('vehicle_sensIT')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/vehicle_sensIT_357.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'electricity':
        print('-'*50)
        print('electricity')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/electricity_44080.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == '2dplanes':
        print('-'*50)
        print('2dplanes_727')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/2dplanes_727.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'creditcard':
        print('-'*50)
        print('creditcard')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/default-of-credit-card-clients_42477.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'nomao':
        print('-'*50)
        print('nomao')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/nomao_1486.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'musk':
        print('-'*50)
        print('musk')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/musk_1116.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'MiniBooNE':
        print('-'*50)
        print('MiniBooNE')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/MiniBooNE_43974.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']         
    elif dataset == 'gas-drift':
        print('-'*50)
        print('gas-drift')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/gas-drift_1476.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
        mask = (target == 1) | (target == 4)
        data = data[mask]
        target = target[mask]
        target = (target == 4).astype(int)        
    else:
        assert False, f"Check {dataset}"
        
    # note
    if dataset != 'gaussian':
        beta_true = None
        assert error_row_rate==None;error_col_rate==None;error_mech==None
    (X, y), (X_val, y_val), (X_test, y_test), error_index, error_row_index, X_original = \
        preprocess_and_split_dataset(data, beta_true, 
                                     experiment, 
                                     target,  n_train, n_val, n_test, 
                                     error_row_rate=error_row_rate,
                                     error_col_rate=error_col_rate,
                                     error_mech=error_mech)   
    
    return (X, y), (X_val, y_val), (X_test, y_test), beta_true, error_index, error_row_index, X_original
  
def preprocess_and_split_dataset(data, beta_true, experiment, target, n_train, n_val, n_test, is_classification=True, 
                                 error_row_rate=0.1,
                                 error_col_rate=0.1,
                                 error_mech='noise'):
    if is_classification is True:
        # classification
        target = target.astype(np.int32)
    else:
        # regression
        raise NotImplementedError('Reg problem not implemented yet!')

    # check constant columns
    constant_columns = np.where(np.std(data, axis=0) == 0)[0]
    data = np.delete(data, constant_columns, axis=1)
    
    ind=np.random.permutation(len(data))
    data, target=data[ind], target[ind]

    data_mean, data_std= np.mean(data, 0), np.std(data, 0)
    data = (data - data_mean) / np.clip(data_std, 1e-12, None)
    n_total=n_train + n_val + n_test

    if len(data) >  n_total:
        X=data[:n_train]
        y=target[:n_train]
        X_val=data[n_train:(n_train+n_val)]
        y_val=target[n_train:(n_train+n_val)]
        X_test=data[(n_train+n_val):(n_train+n_val+n_test)]
        y_test=target[(n_train+n_val):(n_train+n_val+n_test)]
    else:
        assert False, f"Original dataset is less than n_train + n_val + n_test. {len(data)} vs {n_total}. Try again with a smaller number for validation or test."
    
    if experiment == 'outlier':
        X_original = X.copy()
        X, error_index, error_row_index = add_outliers(X, y)
    else:
        error_index = None
        error_row_index = None
        X_original = None
        
    print(f'Train X: {X.shape}')
    print(f'Val X: {X_val.shape}') 
    print(f'Test X: {X_test.shape}') 
    print('-'*30)
    
    return (X, y), (X_val, y_val), (X_test, y_test), error_index, error_row_index, X_original


