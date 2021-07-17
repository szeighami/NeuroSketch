import json
import os

with open('default_conf.json') as f:
    config = json.load(f) 

config['path_to_NeuroDB'] = os.getcwd()
config['exp_name'] = "pm25_RAQ"
config['query_type'] = "RAQ" 

config['data_loc'] = config['path_to_NeuroDB']+"/sample_data/pm25.npy"
config['n'] = 41757 

config['in_dim'] = 3*2
config['out_dim'] = 1
config['db_dim'] = 3 
config['db_ag_col'] = 3 
config['agg_type'] = 1 
config['active_pred_dim'] = 1 


os.system('mkdir tests')
os.system('mkdir tests/'+ config["exp_name"])
with open('tests/'+config["exp_name"]+'/conf.json', 'w') as f:
    json.dump(config, f)

command = 'cd tests/'+ config["exp_name"] + ' && python -u '+ config['path_to_NeuroDB'] +'/main.py conf.json  > out.txt  '
os.system(command)
