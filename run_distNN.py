import json
import os

with open('default_conf.json') as f:
    config = json.load(f) 

config['path_to_NeuroDB'] = os.getcwd()
config['exp_name'] = "gv25_10k_distNN"
config['query_type'] = "dist_NN"

config['data_loc'] = config['path_to_NeuroDB']+"/sample_data/gv25_10k.npy"
config['n'] = 10000
config['in_dim'] = 25
config['db_dim'] = 25

config['k_th'] = 100

os.system('mkdir tests')
os.system('mkdir tests/'+ config["exp_name"])
with open('tests/'+config["exp_name"]+'/conf.json', 'w') as f:
    json.dump(config, f)

command = 'cd tests/'+ config["exp_name"] + ' && python -u '+config['path_to_NeuroDB']+'/main.py conf.json  > out.txt  '
os.system(command)

