import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import spearmanr, rankdata, kendalltau
from utils_eval import ecdf, find_quantile_ecdf
import sage
import shap
import xgboost as xgb

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_values(d,dim):
    values = []
    for i in range(dim):
        values.append(d.get("f%d"%i,0))
    return values

def learn_oob(X_y, oob, global_method = 'SHAP'):
    input_dim = X_y.shape[1] - 1
    
    X_y_train, X_y_test, oob_train, oob_test = train_test_split(X_y, oob, test_size=1000, random_state=0)
    X_y_train, X_y_val, oob_train, oob_val = train_test_split(X_y_train, oob_train, test_size=min(int(0.2 * (X_y.shape[0]-1000)),1000), random_state=0)

    dtrain = xgb.DMatrix(X_y_train, label=oob_train)
    dval = xgb.DMatrix(X_y_val, label=oob_val)
    dtest = xgb.DMatrix(X_y_test, label=oob_test)
    
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
    y_pred = model.predict(dtest)
    score_mse = mean_squared_error(oob_test, y_pred)
    score_mape = mape(oob_test, y_pred)
    assert len(y_pred) == 1000
    
    if global_method == 'SHAP':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_y_test)
        global_importance = np.abs(shap_values.values).mean(axis=0)
        local_importance = np.abs(shap_values.values)
    elif global_method == 'SAGE':
        imputer = sage.MarginalImputer(model, X_y_test[:512])
        estimator = sage.PermutationEstimator(imputer, 'mse')
        global_importance = estimator(X_y_test, oob_test,bar=False).values
    else:
        raise NotImplementedError('Not implemented yet!')
    
    weight_importance = np.array(get_values(model.get_score(importance_type='weight'),input_dim+1))
    gain_importance = np.array(get_values(model.get_score(importance_type='gain'),input_dim+1))
    
    assert len(global_importance) == input_dim + 1
    assert len(weight_importance) == input_dim + 1
    assert len(gain_importance) == input_dim + 1
    
    return {'X_y_split':(X_y_train,X_y_val,X_y_test),
            'oob_split':(oob_train,oob_val,oob_test),
            'score_mse':score_mse,
            'score_mape':score_mape,
            'learn_feature_importance':global_importance[:-1],
            'weight_importance':weight_importance[:-1],
            'gain_importance':gain_importance[:-1],
            'learn_feature_importance_y':global_importance[-1],
            'weight_importance_y':weight_importance[-1],
            'gain_importance_y':gain_importance[-1],
            'learn_feature_importance_y_order':find_quantile_ecdf(global_importance[:-1],global_importance[-1]),
            'weight_importance_y_order':find_quantile_ecdf(weight_importance[:-1],weight_importance[-1]),
            'gain_importance_y_order':find_quantile_ecdf(gain_importance[:-1],gain_importance[-1]),
            'model':model,
            'local_importance':local_importance
           }


def base_learn_oob(X_y_split, global_method = 'SHAP'):
    def split_X_y(data):
        X = data[:,:-1]
        y = data[:,-1]
        return X,y
    
    X_y_train,X_y_val,X_y_test = X_y_split
    input_dim = X_y_train.shape[1] - 1

    base_X_train,base_y_train = split_X_y(X_y_train)
    base_X_val,base_y_val = split_X_y(X_y_val)
    base_X_test,base_y_test = split_X_y(X_y_test)

    base_dtrain = xgb.DMatrix(base_X_train, label=base_y_train)
    base_dval = xgb.DMatrix(base_X_val, label=base_y_val)
    base_dtest = xgb.DMatrix(base_X_test, label=base_y_test)

    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.01,
        'random_state':0
    }

    base_model = xgb.train(
        base_params, base_dtrain, num_boost_round=1000, 
        evals=[(base_dtrain, 'train'), (base_dval, 'eval')], 
        early_stopping_rounds=10, 
        verbose_eval=0
    )
    
    base_y_pred = (base_model.predict(base_dtest) >= 0.5).astype(int)
    acc = accuracy_score(base_y_test, base_y_pred)

    if global_method == 'SHAP':
        base_explainer = shap.TreeExplainer(base_model)
        base_shap_values = base_explainer(base_X_test)
        base_global_importance = np.abs(base_shap_values.values).mean(axis=0)
    elif global_method == 'SAGE':
        base_imputer = sage.MarginalImputer(base_model, base_X_test[:512])
        base_estimator = sage.PermutationEstimator(base_imputer, 'mse')
        base_global_importance = base_estimator(base_X_test, base_y_test,bar=False).values
    else:
        raise NotImplementedError('Not implemented yet!')
        
    base_weight_importance = np.array(get_values(base_model.get_score(importance_type='weight'),input_dim))
    base_gain_importance = np.array(get_values(base_model.get_score(importance_type='gain'),input_dim))
    
    assert len(base_global_importance) == input_dim
    assert len(base_weight_importance) == input_dim
    assert len(base_gain_importance) == input_dim
    
    return {'score_acc':acc, 'learn_feature_importance':base_global_importance,
            'weight_importance':base_weight_importance,'gain_importance':base_gain_importance}

def detect_outlier(arr):
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_small_index = np.where(arr < lower_bound) 
    outlier_large_index = np.where(arr > upper_bound) 
    return outlier_small_index, outlier_large_index

def rcorr(attrA, attrB):
    # rank features (accounting for ties)
    # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
    all_feat_ranksA = rankdata(-np.abs(attrA), method='dense')
    all_feat_ranksB = rankdata(-np.abs(attrB), method='dense')

    return spearmanr(all_feat_ranksA, all_feat_ranksB)[0],kendalltau(all_feat_ranksA, all_feat_ranksB)[0]