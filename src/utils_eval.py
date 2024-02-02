import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_recall_curve, auc, ndcg_score, f1_score
from ensemble_DV_core_original import RandomForestClassifierDV_original


from sklearn.cluster import KMeans
from scipy.integrate import simpson
from scipy.stats import spearmanr, rankdata, ttest_ind, norm, weightedtau
import tqdm

def noisy_detection_experiment(value_dict, noisy_index):
    noisy_score_dict=dict()
    for key in value_dict.keys():
        noisy_score_dict[key]=noisy_detection_core(value_dict[key], noisy_index)

    return noisy_score_dict

def noisy_detection_core(value, noisy_index):
    true_labels = np.zeros_like(value)
    true_labels[noisy_index] = 1
    precision, recall, _ = precision_recall_curve(true_labels, -value)
    return [recall, precision]


def compute_f1_score_by_set(list_a, list_b):
    '''
    Comput F1 score for noisy detection task
    list_a : true flipped data points
    list_b : predicted flipped data points
    '''
    n_a, n_b=len(list_a), len(list_b)
    
    # among A, how many B's are selected
    n_intersection=len(set(list_b).intersection(list_a))
    recall=n_intersection/(n_a+1e-16)
    # among B, how many A's are selected
    precision=n_intersection/(n_b+1e-16)
    
    if recall > 0 and precision > 0:
        f1_score=1/((1/recall + 1/precision)/2)
    else:
        f1_score=0.
    return f1_score

# outlier

def outlier_detection_experiment(value_dict, outlier_inds, X, y):
    outlier_dict = dict()
    for key in value_dict.keys():
        if "treeshap" not in key:  # exclude treeshap
            outlier_dict[key] = outlier_detection_core(outlier_inds, value_dict[key], X, y)
    return outlier_dict

def outlier_detection_core(outlier_inds, df_value, X, y):
    # feat_len/ = X.shape[1]
    detect_rate = []
    count = 0
    sorted_df_value = np.argsort(df_value.flatten())
    for i in range(len(sorted_df_value)):
        cur_ind = sorted_df_value[i]
        if cur_ind in outlier_inds:
            count += 1
        detect_rate.append(count)
    detect_rate = np.array(detect_rate) / detect_rate[-1]
    return detect_rate




# point removal
def point_removal_experiment(value_dict, X, y, X_test, y_test, problem='clf'):
    removal_ascending_dict=dict()
    for key in value_dict.keys():
        removal_ascending_dict[key]=point_removal_core(X, y, X_test, y_test, value_dict[key], ascending=True, problem=problem)
    return {'removal':removal_ascending_dict}

def point_removal_core(X, y, X_test, y_test, value_list, ascending=True, problem='clf'):
    n_sample=len(X)
    if ascending is True:
        sorted_value_list=np.argsort(value_list) # ascending order. low to high.
    
    accuracy_list=[]
    n_period = min(n_sample//100, 5) # we add 1% at each time
    for percentile in tqdm.tqdm(range(0, n_sample // 5+1, n_period)):
        '''
        We repeatedly remove a few of entire data points at each step.
        The data points whose value belongs to the lowest group are removed first.
        The larger, the better
        '''
        sorted_value_list_tmp=sorted_value_list[percentile:]
        if problem == 'clf':
            try:
                clf = LogisticRegression() 
                clf.fit(X[sorted_value_list_tmp], y[sorted_value_list_tmp])
                model_score=clf.score(X_test, y_test)
            except:
                # if y[sorted_value_list_tmp] only has one class
                model_score=np.mean(np.mean(y[sorted_value_list_tmp])==y_test)
        else:
            raise NotImplementedError('Reg problem not implemented yet!')

        accuracy_list.append(model_score)
        
    return accuracy_list


def remove_and_refill(arr, y, index_list):
    # index_list: index to keep
    rows, cols = arr.shape
    
    #remove
    flattened_arr = arr.reshape(-1)
    mask = np.isin(np.arange(rows * cols), index_list)
    result_arr = np.where(mask, flattened_arr, np.nan).reshape(rows, cols)
    
    # remove empty rows
    empty_row = np.where(np.all(np.isnan(result_arr), axis=1))[0]
    result_arr =  np.delete(result_arr,empty_row,axis = 0)
    y = np.delete(y,empty_row,axis = 0)

    #refill
    column_means = np.nanmean(result_arr, axis=0)
    nan_indices = np.isnan(result_arr)
    result_arr[nan_indices] = np.take(column_means, np.where(nan_indices)[1])
    
    # remove empty columns
    empty_column = np.where(np.all(np.isnan(result_arr), axis=0))[0]
    result_arr = np.delete(result_arr,empty_column,axis = 1)
    return result_arr,y,empty_column

# cell fixation & cell removal

def cell_fixation_experiment(value_dict, X, y, X_test, y_test, random=True, X_original=None):
    fixation_dict = dict()
    for key in value_dict.keys():
        fixation_dict[key]=cell_fixation_core(X, y, X_test, y_test, value_list=value_dict[key],ascending=True, random=False, X_original=X_original)
    if random:
        random_array=cell_fixation_core(X, y, X_test, y_test, random=True, X_original=X_original)
        fixation_dict['random']=random_array
    return {'fixation':fixation_dict} 

def cell_fixation_core(X, y, X_test, y_test, value_list=None, ascending=False, problem='clf', random=False, X_original=None):
    n_sample, n_feature = X.shape
    accuracy_list = []

    if random:
        sorted_value_list = np.random.permutation(n_sample*n_feature) 
    else:
        if ascending:
            sorted_value_list = np.argsort(value_list.reshape(-1))
        else:
            sorted_value_list = np.argsort(-value_list.reshape(-1))
    
    n_cell = n_sample * n_feature
    n_period = n_feature

    for percentile in tqdm.tqdm(range(0, n_cell//5 + 1, n_period)):  
        sorted_value_list_tmp = sorted_value_list[:percentile]
        indices_to_replace = np.unravel_index(sorted_value_list_tmp, (n_sample, n_feature))

        X_modified = X.copy()
        X_modified[indices_to_replace] = X_original[indices_to_replace]
        clf = LogisticRegression()  
        clf.fit(X_modified, y)
        model_score = clf.score(X_test, y_test)
        accuracy_list.append(model_score)
        
    return accuracy_list



def cell_removal_experiment(value_dict, X, y, X_test, y_test, random=True):
    removal_dict = dict()
    for key in value_dict.keys():
        if "treeshap" not in key:
            removal_dict[key+"_asc"]=cell_removal_core(X, y, X_test, y_test, value_list=value_dict[key],ascending=True, random=False)
            removal_dict[key+"_des"]=cell_removal_core(X, y, X_test, y_test, value_list=value_dict[key],ascending=False, random=False)
    if random:
        random_array=cell_removal_core(X, y, X_test, y_test, random=True)
        removal_dict['random']=random_array
    return {'removal':removal_dict} 

def cell_removal_core(X, y, X_test, y_test, value_list=None, ascending=False, problem='clf', random=False):
    n_sample, n_feature = X.shape
    accuracy_list = []

    if random:
        sorted_value_list = np.random.permutation(n_sample*n_feature) 
    else:
        if ascending:
            sorted_value_list = np.argsort(value_list.reshape(-1))
        else:
            sorted_value_list = np.argsort(-value_list.reshape(-1))
    
    n_cell = n_sample * n_feature
    n_period = n_feature
    for percentile in tqdm.tqdm(range(0, n_cell//5 + 1, n_period)):  
        sorted_value_list_tmp = sorted_value_list[percentile:]
        clf = LogisticRegression()  
        X_train, y_train, empty_column = remove_and_refill(X,y, sorted_value_list_tmp)
        clf.fit(X_train, y_train)
        if len(empty_column) > 0:
            X_test_tmp = np.delete(X_test,empty_column,axis = 1)
            model_score = clf.score(X_test_tmp, y_test)
        else:
            model_score = clf.score(X_test, y_test)
        accuracy_list.append(model_score)
        
    return accuracy_list


