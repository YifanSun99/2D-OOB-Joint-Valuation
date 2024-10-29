import numpy as np
import time
import pickle
from pathlib import Path
import tqdm

# train_set, train_labels, test_set, test_labels = pickle.load(open('breast_cancer_clean.data', 'rb'))

def knnsv2d_core(train_data, train_labels, test_data, test_labels, K, perm_num):
    
    np.random.seed(perm_num)
    # T = Number of Test Data
    T = len(test_data)
    # N = Number of Train Data
    N = len(train_labels)
    # M = Number of Features
    M = len(train_data[0])
    # 2D SV Matrix
    sv = np.zeros((M,N)) # We will transpose at the end
    feat_count = np.zeros(M)


#     # For each permutation of features # remove this loop, we will loop each permutation separately
#     for i in range(n_perm):

    # Get a random permutation
    perm = np.random.permutation(M) 
#         print("Perm: ", perm[:20])

    for t in range(T): # We will parallelize this loop in different way (technically not needed here)

        y_test = test_labels[t]

        # Feature squared distances from all train points to the current test point
        feat_distances = np.square(train_data - test_data[t]).T # Transpose for easier access

        # Total feature distances from each train point to the current test point
        tot_distances = np.zeros(N)
        
#         shap_time = time.time()
        # Get whether Train Label equals Test Label
        train_2_test_label = (train_labels == y_test).astype(int)
        
        
        # Case: first feature
        feat = perm[0]
        tot_distances = feat_distances[feat]

        rank_rev = np.argsort(-tot_distances) # square root not needed for ranking
        
        # sv1d is shapley values for a given subset of features (not 1D-shapley discusses in paper)
        sv1d = np.zeros(N)
        train_2_test_label_ranked_rev = train_2_test_label[rank_rev]
        
        cur_label = train_2_test_label_ranked_rev[0]
        cur_val = cur_label / N
        
        all_vals = [cur_val]
    
        # we subtract current label with the next label, to see if there are label changes
        # if 0 next data is same class
        # if 1 next data != test class
        # if -1 next data != test class
        train_2_test_label_ranked_diff = train_2_test_label_ranked_rev[1:] - train_2_test_label_ranked_rev[:-1]
        train_2_test_label_ranked_diff_top_K = train_2_test_label_ranked_diff[-K:]
        for label_diff, i in zip(train_2_test_label_ranked_diff, range(N-1,K,-1)): # this will cut before K
            if label_diff:
                cur_val += label_diff / i 
            all_vals.append(cur_val)
        
        for label_diff in train_2_test_label_ranked_diff_top_K: # For Top K
            if label_diff:
                cur_val += label_diff / K
            all_vals.append(cur_val)
            
        sv1d[rank_rev] = all_vals
        sv[perm[1]] -= sv1d

        # the first feature has count 0, because nothing is added
        feat_count += 1
        feat_count[feat] -= 1


        # Case: second to penultimate feature 
        for p in range(1, M-1):
#                 if p % 50 == 0:
#                     print(p)
            feat = perm[p]
            next_feat = perm[p+1]
            
            tot_distances += feat_distances[feat]# to rank
            rank_rev = np.argsort(-tot_distances) # square root not needed for ranking

            sv1d = np.zeros(N)
            train_2_test_label_ranked_rev = train_2_test_label[rank_rev]

            cur_label = train_2_test_label_ranked_rev[0]
            cur_val = cur_label / N

            all_vals = [cur_val]

            train_2_test_label_ranked_diff = train_2_test_label_ranked_rev[1:] - train_2_test_label_ranked_rev[:-1]
            train_2_test_label_ranked_diff_top_K = train_2_test_label_ranked_diff[-K:]
            for label_diff, i in zip(train_2_test_label_ranked_diff, range(N-1,K,-1)): # this will cut before K
                if label_diff:
                    cur_val += label_diff / i 
                all_vals.append(cur_val)

            for label_diff in train_2_test_label_ranked_diff_top_K: # For Top K
                if label_diff:
                    cur_val += label_diff / K
                all_vals.append(cur_val)

            sv1d[rank_rev] = all_vals
            sv[feat] += sv1d
            sv[next_feat] -= sv1d
        
        # Case: last feature
        feat = perm[M-1]
        tot_distances += feat_distances[feat]# to rank
        rank_rev = np.argsort(-tot_distances) # square root not needed for ranking

        sv1d = np.zeros(N)
        train_2_test_label_ranked_rev = train_2_test_label[rank_rev]

        cur_val = train_2_test_label_ranked_rev[0] / N

        all_vals = [cur_val]

        train_2_test_label_ranked_diff = train_2_test_label_ranked_rev[1:] - train_2_test_label_ranked_rev[:-1]
        train_2_test_label_ranked_diff_top_K = train_2_test_label_ranked_diff[-K:]
        for label_diff, i in zip(train_2_test_label_ranked_diff, range(N-1,K,-1)): # this will cut before K
            if label_diff:
                cur_val += label_diff / i 
            all_vals.append(cur_val)

        for label_diff in train_2_test_label_ranked_diff_top_K: # For Top K
            if label_diff:
                cur_val += label_diff / K
            all_vals.append(cur_val)

        sv1d[rank_rev] = all_vals
        sv[feat] += sv1d
        
    return sv, feat_count

def knnsv2d(train_data, train_labels, test_data, test_labels):
    # train_data = train_set
    # train_labels = train_labels
    # test_data = test_set
    # test_labels = test_labels
    n_perm = 1
    K = 10
    
    test_points = len(test_data)
    cycles = 8
    pfrom = 0
    pto = 1000
    
    T = len(test_labels[:test_points])
    N = len(train_labels)
    M = len(train_data[0])
    
    # folder_name = "breast_2d_knn_permutations_clean"
    # Path(folder_name).mkdir(parents=True, exist_ok=True)
    
    sv2d = np.zeros((M, N))
    feat_count = np.zeros(M)
    for perm_num in tqdm.tqdm(range(pfrom, pto)):
        sv = np.zeros((M, N))
        feat_count_tmp = np.zeros(M)
        for i in range(0, test_points, cycles):
            for t in range(i, min(i + cycles, test_points)):
                res = knnsv2d_core(train_data, train_labels, test_data[t:t+1], test_labels[t:t+1], K, perm_num)
                sv += res[0]
                feat_count_tmp += res[1]
        sv2d += sv
        feat_count += feat_count_tmp
    #     pickle.dump(to_save, open(folder_name + "/perm_" + str(perm_num) + ".txt", "wb"))
    norm_sv2d = np.zeros((M, N))
    for i in range(M):
        norm_sv2d[i] = sv2d[i]/feat_count[i]
        
    return norm_sv2d