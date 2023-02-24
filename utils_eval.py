import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.cluster import KMeans
from scipy.integrate import simpson
from scipy.stats import pearsonr, rankdata

'''
noisy detection task
'''

def rankcorr(attrA, attrB, k):
    corrs = []
    # rank features (accounting for ties)
    # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
    all_feat_ranksA = rankdata(-np.abs(attrA), method='ordinal', axis=1)
    all_feat_ranksB = rankdata(-np.abs(attrB), method='ordinal', axis=1)

    for n in range(2,k):
        rho, _ = pearsonr(all_feat_ranksA[all_feat_ranksA <= n], all_feat_ranksB[all_feat_ranksA <= n])
        corrs.append(rho)

    return simpson(corrs, dx = 1)

def noisy_detection_experiment(value_dict, noisy_index):
    noisy_score_dict=dict()
    for key in value_dict.keys():
        noisy_score_dict[key]=noisy_detection_core(value_dict[key], noisy_index)

    noisy_dict={'Meta_Data': ['Recall', 'Kmeans_label'],
                'Results': noisy_score_dict}
    return noisy_dict

def masked_detection_experiment(value_dict, masked_index):
    masked_score_dict=dict()
    for key in value_dict.keys():
        masked_score_dict[key]=noisy_detection_core(np.abs(value_dict[key]), masked_index)

    masked_dict={'Meta_Data': ['Recall', 'Kmeans_label'],
                'Results': masked_score_dict}
    return masked_dict

def corr_experiment(value_dict, beta_true):
    corr_score_dict=dict()
    for key in value_dict.keys():
        corr_score_dict[key]=corr_evaluation_core(value_dict[key], beta_true)
    corr_dict={'Meta_Data': ['Corr'],
                'Results': corr_score_dict}
    return corr_dict

def corr_evaluation_core(value, beta_true):
    attrA = np.abs(beta_true).reshape(1, -1)
    attrB = np.abs(value).reshape(1, -1)
    return rankcorr(attrA, attrB, k = (beta_true != 0).sum())

def noisy_detection_core(value, noisy_index):
    # without kmeans algorithm (but requires prior knowledge of the number of noise labels)
    index_of_small_values=np.argsort(value)[:len(noisy_index)]
    recall=len([ind for ind in index_of_small_values if ind in noisy_index])/len(noisy_index)
    
    # using kmeans label
    kmeans=KMeans(n_clusters=2, random_state=0).fit(value.reshape(-1, 1))
    guess_index=np.where(kmeans.labels_ == np.argmin(kmeans.cluster_centers_))[0]
    f1_kmeans_label=compute_f1_score_by_set(noisy_index, guess_index)

    return [recall, f1_kmeans_label] 

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
    
def point_removal_experiment(value_dict, X, y, X_test, y_test, problem='clf'):
    removal_ascending_dict, removal_descending_dict=dict(), dict()
    for key in value_dict.keys():
        removal_ascending_dict[key]=point_removal_core(X, y, X_test, y_test, value_dict[key], ascending=True, problem=problem)
        removal_descending_dict[key]=point_removal_core(X, y, X_test, y_test, value_dict[key], ascending=False, problem=problem)
    random_array=point_removal_core(X, y, X_test, y_test, 'Random', problem=problem)
    removal_ascending_dict['Random']=random_array
    removal_descending_dict['Random']=random_array
    return {'ascending↑':removal_ascending_dict, 'descending↓':removal_descending_dict}

def point_removal_core(X, y, X_test, y_test, value_list, ascending=True, problem='clf'):
    n_sample=len(X)
    if value_list == 'Random':
        sorted_value_list=np.random.permutation(n_sample) 
    else:
        if ascending is True:
            sorted_value_list=np.argsort(value_list) # ascending order. low to high.
        else:
            sorted_value_list=np.argsort(value_list)[::-1] # descending order. high to low.
    
    accuracy_list=[]
    n_period = min(n_sample//100, 5) # we add 1% at each time
    for percentile in range(0, n_sample, n_period):
        '''
        We repeatedly remove 5% of entire data points at each step.
        The data points whose value belongs to the lowest group are removed first.
        The larger, the better
        '''
        sorted_value_list_tmp=sorted_value_list[percentile:]
        if problem == 'clf':
            try:
                clf=LogisticRegression() 
                clf.fit(X[sorted_value_list_tmp], y[sorted_value_list_tmp])
                model_score=clf.score(X_test, y_test)
            except:
                # if y[sorted_value_list_tmp] only has one class
                model_score=np.mean(np.mean(y[sorted_value_list_tmp])==y_test)
        else:
            try:
                model=LinearRegression() 
                model.fit(X[sorted_value_list_tmp], y[sorted_value_list_tmp])
                model_score=model.score(X_test, y_test)
            except:
                # if y[sorted_value_list_tmp] only has one class
                model_score=0

        accuracy_list.append(model_score)
        
    return simpson(accuracy_list[:int(len(accuracy_list)/5)], dx = n_period/100)

