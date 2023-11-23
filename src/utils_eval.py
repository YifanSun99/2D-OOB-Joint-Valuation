import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_recall_curve, auc, ndcg_score, f1_score
from ensemble_DV_core_original import RandomForestClassifierDV_original


from sklearn.cluster import KMeans
from scipy.integrate import simpson
from scipy.stats import spearmanr, rankdata, ttest_ind, norm, weightedtau
import tqdm

# noisy (not included in the paper)
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

# def noisy_detection_core(value, noisy_index):
#     # without kmeans algorithm (but requires prior knowledge of the number of noise labels)
#     index_of_small_values=np.argsort(value)[:len(noisy_index)]
#     recall=len([ind for ind in index_of_small_values if ind in noisy_index])/len(noisy_index)
    
#     # using kmeans label
#     kmeans=KMeans(n_clusters=2, random_state=0, n_init='auto').fit(value.reshape(-1, 1))
#     guess_index=np.where(kmeans.labels_ == np.argmin(kmeans.cluster_centers_))[0]
#     f1_kmeans_label=compute_f1_score_by_set(noisy_index, guess_index)

#     return [recall, f1_kmeans_label] 

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

# NOTE: class-wise detection below
# def outlier_detection_experiment(value_dict, outlier_inds_0, outlier_inds_1, X, y):
#     outlier_dict=dict()
#     for key in value_dict.keys():
#         if "treeshap" not in key: #exclude treeshap
#             outlier_dict[key+"_0"]=outlier_detection_core(outlier_inds_0, 0, value_dict[key], X, y)
#             outlier_dict[key+"_1"]=outlier_detection_core(outlier_inds_1, 1, value_dict[key], X, y)

#     return outlier_dict

# def outlier_detection_core(outlier_inds_class, cur_class, df_value, X, y):
#     feat_len = X.shape[1]
#     class_inds = np.where(y == cur_class)[0]
#     class_pert_inds = []
#     for i in outlier_inds_class:
#         r = i//feat_len
#         c = i % feat_len
#         pos_s = np.argwhere(class_inds == r)[0,0]
#         pos_f = c
#         new_ind = pos_s * feat_len + pos_f
#         class_pert_inds.append(new_ind)
#     detect_rate = []
#     count = 0
#     sorted_df_value = np.argsort(df_value[class_inds,:].flatten())
#     for i in range(len(sorted_df_value)):
#         cur_ind = sorted_df_value[i]
#         if cur_ind in class_pert_inds:
#             count += 1
#         detect_rate.append(count)
#     detect_rate = np.array(detect_rate) / detect_rate[-1]
#     return detect_rate

# error
def kmeans_smaller_cluster_indices_row(matrix, error_row_index):
    binary_matrix = np.zeros_like(matrix)
    if isinstance(error_row_index,np.ndarray):
        to_enumerate = error_row_index
    else:
        raise ValueError('You should input error_row_index')
    for i in to_enumerate:
        row = matrix[i, :]
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(row.reshape(-1, 1))
        smaller_cluster_idx = np.where(kmeans.labels_ == np.argmin(kmeans.cluster_centers_))[0]
        binary_matrix[i, smaller_cluster_idx] = 1
    return binary_matrix

def get_min_k_indices(matrix, k, error_row_index): 
    binary_matrix = np.zeros_like(matrix)
    if isinstance(error_row_index,np.ndarray):
        to_enumerate = error_row_index
    else:
        raise ValueError('You should input error_row_index')
        # to_enumerate = range(matrix.shape[0])
    for i in to_enumerate:
        row_data = matrix[i]
        sorted_indices = np.argsort(row_data)
        binary_matrix[i, sorted_indices[:k]] = 1
    return binary_matrix


def error_detection_experiment(df_value_dict, error_index, error_row_index,
                               two_stage = True, data_value_dict = None):
    error_score_dict=dict()
    if not two_stage:
        for key in df_value_dict.keys():
            error_score_dict[key]=error_detection_core(df_value_dict[key], error_index, error_row_index, method='min_k')
    else:
        raise NotImplementedError('Not implemented yet!')
    
    #prepare random baseline
    random_guess = np.zeros(error_index.shape)
    for i in error_row_index:
        random_guess[i] = np.random.binomial(1,0.5,error_index.shape[1])
    
    error_score_dict['random'] = [f1_by_row(error_index, random_guess, error_row_index)]
    
    error_dict={'Meta_Data': ['min_k'],
                'Results': error_score_dict}
    return error_dict
    
def error_detection_core(value, error_index, error_row_index, method='k_means'):
    # error_rate = sum(error_index[0])/len(error_index[0])
    if method == 'min_k':
        prediction = get_min_k_indices(value, max(error_index.sum(axis=1)), error_row_index)
    elif method == 'k_means':
        prediction = kmeans_smaller_cluster_indices_row(value, error_row_index)
    else:
        raise NotImplementedError('Not implemented yet!')
    
    # using kmeans label
    score = f1_by_row(error_index, prediction, error_row_index)
    return [score]    

def f1_by_row(true, pred, row_index):
    return np.mean([f1_score(true[ind], pred[ind], average='binary') for 
                    ind in row_index])

# point removal
def point_removal_experiment(value_dict, X, y, X_test, y_test, problem='clf'):
    removal_ascending_dict=dict()
    for key in value_dict.keys():
        removal_ascending_dict[key]=point_removal_core(X, y, X_test, y_test, value_dict[key], ascending=True, problem=problem)
    random_array=point_removal_core(X, y, X_test, y_test, 'random', problem=problem)
    removal_ascending_dict['random']=random_array
    return {'removal':removal_ascending_dict}

def point_removal_core(X, y, X_test, y_test, value_list, ascending=True, problem='clf'):
    n_sample=len(X)
    if value_list == 'random':
        sorted_value_list=np.random.permutation(n_sample) 
    else:
        if ascending is True:
            sorted_value_list=np.argsort(value_list) # ascending order. low to high.
    
    accuracy_list=[]
    n_period = min(n_sample//100, 5) # we add 1% at each time
    for percentile in tqdm.tqdm(range(0, n_sample//2+1, n_period)):
        '''
        We repeatedly remove a few of entire data points at each step.
        The data points whose value belongs to the lowest group are removed first.
        The larger, the better
        '''
        sorted_value_list_tmp=sorted_value_list[percentile:]
        if problem == 'clf':
            try:
                # clf=RandomForestClassifierDV_original(n_estimators=2000, n_jobs=-1)
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

#feature removal
def feature_removal_experiment(value_dict, X, y, X_test, y_test, random=True):
    removal_dict = dict()
    for key in value_dict.keys():
        if "treeshap" not in key:
            removal_dict[key]=feature_removal_core(X, y, X_test, y_test, np.abs(value_dict[key]))
        else:
            removal_dict[key]=feature_removal_core(X, y, X_test, y_test, value_dict[key])
    if random:
        random_array=feature_removal_core(X, y, X_test, y_test, 'random')
        removal_dict['random']=random_array
    return {'removal':removal_dict} #the smaller the better

def feature_removal_core(X, y, X_test, y_test, value_list):
    d_sample=X.shape[1]
    if value_list == 'random':
        sorted_value_list=np.random.permutation(d_sample) 
    else:
        sorted_value_list=np.argsort(-value_list) # descending order. high to low.
    
    removal_dim = d_sample#max(d_sample//2,5)
    accuracy_list=[]
    for n_remove in tqdm.tqdm(range(removal_dim)):
        sorted_value_list_tmp=sorted_value_list[n_remove:]
        # clf=RandomForestClassifierDV_original(n_estimators=2000, n_jobs=-1)
        clf = LogisticRegression() 
        clf.fit(X[:,sorted_value_list_tmp], y)
        model_score=clf.score(X_test[:,sorted_value_list_tmp], y_test)

        accuracy_list.append(model_score)
        
    return accuracy_list#,simpson(accuracy_list, dx = 1),simpson(accuracy_list[:max(5,int(len(accuracy_list)//5))], dx = 1)


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

# cell removal

def cell_removal_experiment(value_dict, X, y, X_test, y_test, random=True):
    removal_dict = dict()
    for key in value_dict.keys():
        if "treeshap" not in key:
            removal_dict[key+"_asc"]=cell_removal_core(X, y, X_test, y_test, value_dict[key],ascending=True)
            removal_dict[key+"_des"]=cell_removal_core(X, y, X_test, y_test, value_dict[key],ascending=False)
    if random:
        random_array=cell_removal_core(X, y, X_test, y_test, 'random')
        removal_dict['random']=random_array
    return {'removal':removal_dict} 

def cell_removal_core(X, y, X_test, y_test, value_list, ascending=False, problem='clf'):
    n_sample, n_feature = X.shape
    accuracy_list = []

    if value_list == 'random':
        sorted_value_list = np.random.permutation(n_sample*n_feature) 
    else:
        if ascending:
            sorted_value_list = np.argsort(value_list.reshape(-1))
        else:
            sorted_value_list = np.argsort(-value_list.reshape(-1))
    
    n_cell = n_sample * n_feature
    n_period = n_feature
    for percentile in tqdm.tqdm(range(0, n_cell//2, n_period)):  
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


