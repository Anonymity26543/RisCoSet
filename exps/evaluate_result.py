import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from typing import List
import ipdb
import random

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)
from utils import utils

from scipy.stats import binom, norm


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
   
 
def hb_p_value(r_hat, n, alpha):
    bentkus_p_value = np.e*binom.cdf(np.ceil(n*r_hat),n,alpha)
    return bentkus_p_value



if __name__ == "__main__":
    seed = 2026
    set_seed(seed)
    
    # FROM DATA METHOD
    dataset = "HumanEval"
    # dataset = "MBPP"
    # dataset = "APPS"
    d = 0.1
    m = 1
    s = 20
    calib_ratio = 0.5
    # computed = []

    data = []
    try:
        print(f"{ROOT_DIR}/output/results/{dataset}/output__m_{m}__s_{s}.json")
        data += utils.read_json(
            f"{ROOT_DIR}/output/results/{dataset}/output__m_{m}__s_{s}.json"
        )["output"]
    except:
        pass
    print("retrieved data", len(data))
    
    map_tau_to_target_in_pruned_pred = {}
    map_tau_to_frac_rm = {}
    for sample in data:
        # print(sample["output"].keys())
        # print(sample["output"]["frac_included"])
        # tau = np.exp(-1 * sample["cost"])
        Lambda = sample["cost"]
        if Lambda not in map_tau_to_target_in_pruned_pred:
            map_tau_to_target_in_pruned_pred[Lambda] = []
            map_tau_to_frac_rm[Lambda] = []

        map_tau_to_target_in_pruned_pred[Lambda].append(sample["pred_in_target"]["eval"])
        map_tau_to_frac_rm[Lambda].append(sample["output"]["frac_included"])
    
    assert len(data) % len(map_tau_to_frac_rm) == 0
    n = int(len(data) / len(map_tau_to_frac_rm))
    n_calib = int(n * calib_ratio)

    # map data to e space
    alpha = np.linspace(0.05, 0.25, num = 5).tolist()
    print("alpha: ", alpha)
    target_coverage = []
     
    
    bc_alpha_to_lambda_lst_all = []
    bc_alpha_to_coverage_lst_all = []
    bc_alpha_to_frac_rm_lst_all = []
    
    hb_alpha_to_lambda_lst_all = []
    hb_alpha_to_coverage_lst_all = []
    hb_alpha_to_frac_rm_lst_all = []
    
    fst_alpha_to_lambda_lst_all = []
    fst_alpha_to_coverage_lst_all = []
    fst_alpha_to_frac_rm_lst_all = []
    for _ in range(100):
        sample_idx = list(np.arange(n))
        calib_idx = random.sample(sample_idx, n_calib)
        test_idx = list(set(sample_idx) - set(calib_idx))
        
        # construct calibration/test set
        map_tau_to_target_in_pruned_pred_calib, map_tau_to_target_in_pruned_pred_test = {}, {}
        map_tau_to_frac_rm_calib, map_tau_to_frac_rm_test = {}, {}
        
        
        lambda_lst = []
        coverage_lst_calib, coverage_lst_test = [], []
        frac_rm_lst_calib, frac_rm_lst_test = [], []
        for key in map_tau_to_target_in_pruned_pred:
            lambda_lst.append(key)
            
            # map_tau_to_target_in_pruned_pred
            cov_arr = np.array(map_tau_to_target_in_pruned_pred[key])
            map_tau_to_target_in_pruned_pred_calib[key] = list(cov_arr[calib_idx])
            map_tau_to_target_in_pruned_pred_test[key] = list(cov_arr[test_idx])
            
            coverage_lst_calib.append(np.sum(map_tau_to_target_in_pruned_pred_calib[key]) / len(map_tau_to_target_in_pruned_pred_calib[key]) * 100)
            coverage_lst_test.append(np.sum(map_tau_to_target_in_pruned_pred_test[key]) / len(map_tau_to_target_in_pruned_pred_test[key]) * 100)

            # map_tau_to_frac_rm
            frac_arr = np.array(map_tau_to_frac_rm[key])
            map_tau_to_frac_rm_calib[key] = list(frac_arr[calib_idx])
            map_tau_to_frac_rm_test[key] = list(frac_arr[test_idx])
            
            frac_rm_lst_calib.append(100 - np.mean(map_tau_to_frac_rm_calib[key]) * 100)
            frac_rm_lst_test.append(100 - np.mean(map_tau_to_frac_rm_test[key]) * 100)

        assert len(lambda_lst) == len(coverage_lst_calib)
        assert len(lambda_lst) == len(coverage_lst_test)
        assert len(lambda_lst) == len(frac_rm_lst_calib)
        assert len(lambda_lst) == len(frac_rm_lst_test)
        # print(coverage_lst)
        # print(frac_rm_lst)

        # Bonferroni correction
        index1 = []
        alpha_to_lambda_lst = []
        alpha_to_coverage_lst = []
        alpha_to_frac_rm_lst = []
        for i in range(len(alpha)):
            alpha_to_lambda_index = -1
            for j in range(len(lambda_lst)):
                p = hb_p_value(1 - coverage_lst_calib[j]*0.01, n_calib, alpha[i])
                if p <= d / len(lambda_lst):
                    index1.append(j)
                    alpha_to_lambda_index = j
                    alpha_to_lambda_lst.append(lambda_lst[alpha_to_lambda_index])
                    alpha_to_coverage_lst.append(coverage_lst_test[alpha_to_lambda_index])
                    alpha_to_frac_rm_lst.append(frac_rm_lst_test[alpha_to_lambda_index])
                    break
                
            if alpha_to_lambda_index == -1:
                print('Error: not find coverage')
                exit(0) 
                
            # best_index = alpha_to_lambda_index[0]
            # for index in alpha_to_lambda_index:
            #     if frac_rm_lst_calib[index] < frac_rm_lst_calib[best_index]:
            #         best_index = index
        
        bc_alpha_to_coverage_lst_all.append(alpha_to_coverage_lst)
        bc_alpha_to_frac_rm_lst_all.append(alpha_to_frac_rm_lst)

        # Holm–Bonferroni
        index2 = []
        alpha_to_lambda_lst = []
        alpha_to_coverage_lst = []
        alpha_to_frac_rm_lst = []
        for i in range(len(alpha)):
            alpha_to_lambda_index = []
            p_list = []
            thres_list = []
            for j in range(len(lambda_lst)):
                p = hb_p_value(1 - coverage_lst_calib[j]*0.01, n_calib, alpha[i])
                p_list.append(p)
                thres_list.append(d / (len(lambda_lst) - j))
            
            for j in range(len(thres_list)):
                min_p = np.min(p_list)
                min_p_index = np.argmin(p_list)
                if min_p <= thres_list[j]:
                    alpha_to_lambda_index.append(min_p_index)
                    p_list[min_p_index] = 10000
                    continue
                else:
                    break
                
            if len(alpha_to_lambda_index) == 0:
                print('Error: not find coverage')
                exit(0) 
                
            best_index = min(alpha_to_lambda_index)
            index2.append(best_index)
                

            alpha_to_lambda_lst.append(lambda_lst[best_index])
            alpha_to_coverage_lst.append(coverage_lst_test[best_index])
            alpha_to_frac_rm_lst.append(frac_rm_lst_test[best_index])
        
        hb_alpha_to_coverage_lst_all.append(alpha_to_coverage_lst)
        hb_alpha_to_frac_rm_lst_all.append(alpha_to_frac_rm_lst)
        
        # Fixed sequence testing
        index3 = []
        
        alpha_to_lambda_lst = []
        alpha_to_coverage_lst = []
        alpha_to_frac_rm_lst = []
        m = 1.0
        # t = len(lambda_lst) / m
        for i in range(len(alpha)):
            alpha_to_lambda_index = []
            for j in range(len(lambda_lst)):
                p = hb_p_value(1 - coverage_lst_calib[len(lambda_lst)-j-1]*0.01, n_calib, alpha[i])
                if p <= d / m:
                    alpha_to_lambda_index.append(len(lambda_lst)-j-1)
                    continue
                else:
                    break
                
            if alpha_to_lambda_index == -1:
                print('Error: not find coverage')
                exit(0) 
                
            best_index = min(alpha_to_lambda_index)
            index3.append(best_index)
            
            alpha_to_lambda_lst.append(lambda_lst[best_index])
            alpha_to_coverage_lst.append(coverage_lst_test[best_index])
            alpha_to_frac_rm_lst.append(frac_rm_lst_test[best_index])
        
        fst_alpha_to_coverage_lst_all.append(alpha_to_coverage_lst)
        fst_alpha_to_frac_rm_lst_all.append(alpha_to_frac_rm_lst)
        
        # print(index1)
        # print(index2)
        # print(index3)



    # print(alpha_to_lambda_lst_all)
    print("======Bonferroni correction======")
    print(list(np.mean(bc_alpha_to_coverage_lst_all, axis=0)))
    print(list(np.mean(bc_alpha_to_frac_rm_lst_all, axis=0)))
    
    print("======Holm–Bonferroni======")
    print(list(np.mean(hb_alpha_to_coverage_lst_all, axis=0)))
    print(list(np.std(hb_alpha_to_coverage_lst_all, axis=0)))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(list(np.mean(hb_alpha_to_frac_rm_lst_all, axis=0)))
    print(list(np.std(hb_alpha_to_frac_rm_lst_all, axis=0)))
    
    print("======Fixed sequence testing======")
    print(list(np.mean(fst_alpha_to_coverage_lst_all, axis=0)))
    print(list(np.std(fst_alpha_to_coverage_lst_all, axis=0)))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(list(np.mean(fst_alpha_to_frac_rm_lst_all, axis=0)))
    print(list(np.std(fst_alpha_to_frac_rm_lst_all, axis=0)))

    
    