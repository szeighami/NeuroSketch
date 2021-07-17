import json

config={}
config['path_to_NeuroDB'] = "" #absolute path to NeuroDB files
config['exp_name'] = "" #test results are written in folder path_to_NeuroDB/tests/exp_name
config['query_type'] = "" #supported types are "RAQ", "dist_NN" and "NN" 

config['data_loc'] = "" #path to the dataset. Must be a .npy file
config['n'] = 10000 #total data size

#Neural network specifications
config['in_dim'] = 3*2 #input dimensionality of the neural network
config['out_dim'] = 1 #output dimensionality of the neural network, should be equal to in_dim for NN query type
config['depth'] = 4 #height of the kd-tree
config['no_filters'] = 2 #branching factor for kdtree
config['filter_width1'] = 60 # number of units in the first layer of the neural network
config['filter_width2'] = 30 # number of units in the remaining layers of the neural network
config['phi_no_layers'] = 5 # number of layers of the neural network
config['lr']=0.001#initial learning rate
config['min_lr']=0.0001#final learning rate, after decay
config['batch_size'] = 50 #number of batches per epoch (NOT size of the batch)
config['EPOCHS'] = 100 # number of epochs
config['print_freq'] = 100 # Frequency to print training statistics
config['train_size'] = 100000 #No queries for training
config['test_size'] = 10000 #No queries for testing
config['train_data_size'] = 0#size of dataset used for training. A random subset is sampled if less than n but more than 0

#workload specification for RAQs
config['db_dim'] = 3 #number of attributes in the predicate for RAQs
config['db_ag_col'] = 3 #measure for RAQs
config['q_range'] = 1 #proportion of query space to use for each range predicate. q_range=1 means no restriction
config['agg_type'] = 1 #betwee 0-3, means 1:avg, 0:std, 2: count, 3:sum
config['with_angle'] = False #to generate median visit duration with general rectangle queries
config['active_pred_dim'] = 1 #no. active attributes
config['active_pred_dim_pairs'] = 10 #if active_pred_dim!=1, no. different active attribut pairs to use
config['MAX_VAL'] = 10 #normaliation range for RAQs

#workload specification for NN/dist_NN queries
config['k_th'] = 100

#training parallelization
config['no_processes'] = 1 #number of different neural networks to train simultaneously
config["no_threads"] =20 #number of threads when collecting answers to training queries. train_size should be divisible by no_threads

with open('default_conf.json', 'w') as f:
    json.dump(config, f)
