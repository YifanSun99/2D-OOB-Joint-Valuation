import numpy as np
import time
from numpy import random
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tqdm
import pickle

nums = 10
def calc_perf(pos, feat_len, train, test, data_perm, feat_perm):
#     if pos % 10000 == 0:
#         print(pos)
    i = int(pos / feat_len)
    j = int(pos % feat_len)
    data_i = data_perm[i] # curr data
    subset_data_i = data_perm[:i+1] # data indices including i

    # get data including i
    sub_train_i = train[subset_data_i, :]
    
    feat_j = feat_perm[j] # curr feature
    subset_feat_j = feat_perm[:j+1] # feature indices including j (= removing features after j)

    ## i and j
    sub_train_i_j = sub_train_i[:, subset_feat_j]

    DC = DecisionTreeClassifier()
    
    acc_i_j = 0
    
    for _ in range(nums):
        DC.fit(sub_train_i_j, sub_train_i[:, feat_len]) # sub_train_i[:,feat_len] - labels
        pred = DC.predict(test[:, subset_feat_j])
        acc_i_j += accuracy_score(test[:, feat_len], pred)

    acc_i_j /= nums
    return acc_i_j




def mcsv2d(train_data, train_labels, test_data, test_labels):
    train = np.hstack((train_data, train_labels.reshape(-1, 1)))
    test = np.hstack((test_data, test_labels.reshape(-1, 1)))
    train_len = len(train)
    feat_len = train.shape[1]-1

    train_arr = np.arange(len(train))
    feat_arr = np.arange(train.shape[1]-1)

        
    # values of cells in the matrix, initialized to 0\n",
    # will keep a 2D array for faster changes rather than a 2D dictionary\n",
    cells = np.zeros((len(train_arr), len(feat_arr)))

    verbose = False
    pfrom = 0
    print(f"p from {pfrom}")
    pto = 15
    print(f"p to {pto}")

    perms = range(pfrom, pto)
    print(f"perms: {perms}")

    sum_vals_1 = np.zeros((train_len, feat_len))

    for p in perms:
        print("p: ", p)
        # get a data permutation\n",

        data_perm_time = time.time()
        data_perm = random.permutation(train_arr)
        data_perm_time = time.time() - data_perm_time

        # get a feature permutation\n",
        feat_perm_time = time.time()
        feat_perm = random.permutation(feat_arr)
        feat_perm_time = time.time() - feat_perm_time
        saved_values = np.zeros(feat_len)
        last_value = 0

        start_time = time.perf_counter()
        result = [calc_perf(pos, feat_len, train, test, data_perm, feat_perm) for pos in tqdm.tqdm(range(0, train_len * feat_len))]
        finish_time = time.perf_counter()
        print(f"Program finished in {finish_time-start_time} seconds")

        model_perfs = np.zeros((train_len, feat_len))
        for i in range(len(result)):
            model_perfs[int(i/feat_len)][i%feat_len] = result[i]
        vals_1 = np.zeros((train_len, feat_len))
        
        for position in range(len(result)):

            i = int(position / feat_len)
            j = int(position % feat_len)

            val00 = 0
            val01 = 0
            val10 = 0
            if i > 0:
                val01 = model_perfs[i-1][j]
                if j > 0:
                    val00 = model_perfs[i-1][j-1]
            if j > 0:
                val10 = model_perfs[i][j-1]

            val11 = model_perfs[i][j]

            cur_perf = model_perfs[i][j] + val00 - val01 - val10

            data_i = data_perm[i] # curr data 
            feat_j = feat_perm[j] # curr feature

            vals_1[data_i][feat_j] = cur_perf
        
        sum_vals_1 += vals_1
    average_vals_1 = sum_vals_1 / len(perms)    
    return average_vals_1