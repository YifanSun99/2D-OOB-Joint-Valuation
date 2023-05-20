import numpy as np
import pandas as pd
import pickle
from scipy.stats import t


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
    
        
        if experiment == 'noisy':
            # no normalization
            # beta
            beta_true = np.random.normal(size=(input_dim,1))
            # no mask
        elif experiment == 'error':
            # normalization
            data_mean, data_std= np.mean(data, 0), np.std(data, 0)
            data = (data - data_mean) / np.clip(data_std, 1e-12, None)
            # beta
            beta_true = np.random.normal(loc = 1,scale = np.sqrt(0.1), size=(input_dim,1))
            # no mask
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
    elif dataset == 'webdata_wXa':
        print('-'*50)
        print('webdata_wXa')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/webdata_wXa_350.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'MiniBooNE':
        print('-'*50)
        print('MiniBooNE')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/MiniBooNE_43974.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']         
    else:
        assert False, f"Check {dataset}"
        
    # note
    if dataset != 'gaussian':
        beta_true = None
    (X, y), (X_val, y_val), (X_test, y_test), error_index, error_row_index, X_original = \
        preprocess_and_split_dataset(data, beta_true, 
                                     experiment, 
                                     target,  n_train, n_val, n_test, 
                                     error_row_rate=error_row_rate,
                                     error_col_rate=error_col_rate,
                                     error_mech=error_mech)   
    
    if dataset == 'gaussian':
        return (X, y), (X_val, y_val), (X_test, y_test), beta_true, error_index, error_row_index, X_original
    else:
        return (X, y), (X_val, y_val), (X_test, y_test), None, None, None, None
  
def preprocess_and_split_dataset(data, beta_true, experiment, target, n_train, n_val, n_test, is_classification=True, 
                                 error_row_rate=0.1,
                                 error_col_rate=0.1,
                                 error_mech='noise'):
    if is_classification is True:
        # classification
        target = target.astype(np.int32)
    else:
        # regression
        target_mean, target_std= np.mean(target, 0), np.std(target, 0)
        target = (target - target_mean) / np.clip(target_std, 1e-12, None)
    
    ind=np.random.permutation(len(data))
    data, target=data[ind], target[ind]

    data_mean, data_std= np.mean(data, 0), np.std(data, 0)
    data = (data - data_mean) / np.clip(data_std, 1e-12, None)
    n_total=n_train + n_val + n_test
    # print(data.mean(axis=0),data.std(axis=0))

    if len(data) >  n_total:
        X=data[:n_train]
        y=target[:n_train]
        X_val=data[n_train:(n_train+n_val)]
        y_val=target[n_train:(n_train+n_val)]
        X_test=data[(n_train+n_val):(n_train+n_val+n_test)]
        y_test=target[(n_train+n_val):(n_train+n_val+n_test)]
    else:
        assert False, f"Original dataset is less than n_train + n_val + n_test. {len(data)} vs {n_total}. Try again with a smaller number for validation or test."
    
    if experiment == 'error':
        # observe data with measurement error (note this is only done for training data)
        if error_mech == 'noise':
            # error = np.random.normal(scale=1,size=X.shape)
            error = t.rvs(df=1, size=X.shape)
        elif error_mech == 'adv':
            eta = 5
            c = int(error_col_rate*X.shape[1])
            inner_dot = X.dot(beta_true)
            error = np.tile(-inner_dot / c - np.sign(y.reshape(-1,1) - 0.5) * eta / c,(1,X.shape[1]))
            assert error.shape == X.shape
    
        # by row
        error_indicator = np.zeros(X.shape, dtype=int)
        error_row_index = np.sort(np.random.choice(X.shape[0], int(error_row_rate*X.shape[0]), replace=False))
        for i in error_row_index:
            random_indices = np.random.choice(X.shape[1], int(error_col_rate*X.shape[1]), replace=False)
            error_indicator[i, random_indices] = 1
    
        error_index = error_indicator
        X_original = X.copy()
        X += error*error_indicator
        
    else:
        error_index = None
        error_row_index = None
        X_original = None
        
    print(f'Train X: {X.shape}')
    print(f'Val X: {X_val.shape}') 
    print(f'Test X: {X_test.shape}') 
    print('-'*30)
    
    return (X, y), (X_val, y_val), (X_test, y_test), error_index, error_row_index, X_original


