import subprocess
import os
from kd_tree import build_tree
import numpy as np


def call_tree(config, DB, queries, test_queries, res, test_res):
    processes = []
    min_dims = np.zeros(config['in_dim'])-0.5*config['MAX_VAL']
    max_dims = np.zeros(config['in_dim'])+0.5*config['MAX_VAL']
    in_DB = None
    in_res = res
    in_test_res = test_res
    in_test_res = in_test_res.reshape((-1, config['out_dim']))
    in_res = in_res.reshape((-1, config['out_dim']))

    my_model, _, processes = build_tree(config['depth'], config['no_filters'], config['in_dim'], queries, in_res, test_queries, in_test_res, processes, config['exp_name'], "", 0, 0, config['no_processes'], config['path_to_NeuroDB'])

    if len(processes) > 0:
        for p in processes:
            p.wait()
    return my_model

def test_model(config, test_qs, test_res):
    np.savetxt(config['exp_name']+"_queries.txt", test_qs, delimiter=",")
    np.savetxt(config['exp_name']+"_res.txt", test_res, delimiter=",")
    c_call = config["path_to_NeuroDB"]+"/run_model "+str(config['no_filters'])+" " + str(config['in_dim'])+" " + str(config['out_dim']) +' ' + str(config['depth'])+ ' ' + config['exp_name'] + ' ' + config['exp_name'] + " 10000 1"
    print("Testing model")
    os.system(c_call)

def save_model(config, my_model):
    if config['no_filters'] == 1:
        my_model.save_params(config['exp_name']+'.m')
    else:
        cnt = my_model.get_params(config['exp_name'], '')
        with open(config['exp_name']+'_tree.m', 'w') as f:
           f.write(cnt) 


