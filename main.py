import time
import sys
import json
from model_utils import call_tree, save_model, test_model
from data_utils import get_nn_data, get_raq_data


with open(sys.argv[1], 'r') as f:
    config = json.load(f) 

start = time.time()

print("Starting data collection")
if config["query_type"] == "NN" or config["query_type"] == "dist_NN":

    DB, queries, test_queries, res, test_res = get_nn_data(config["n"], config["in_dim"], config["query_type"] == "dist_NN", config["train_size"], config["test_size"], config["k_th"], config["data_loc"], config["MAX_VAL"], config["no_threads"])
elif config["query_type"] == "RAQ":

    DB, queries, test_queries, res, test_res = get_raq_data(config["n"], config["db_dim"], config["in_dim"], config["train_size"], config["test_size"], config['q_range'], config['agg_type'], config['MAX_VAL'], config['data_loc'], config['db_ag_col'], config["train_data_size"], config["active_pred_dim"], config["active_pred_dim_pairs"], config["with_angle"], config["no_threads"])
else:
    raise ValueError("ValueError exception thrown")


end = time.time()
print("Data collection took {:.2f}s".format(end-start))

start = time.time()

my_model = call_tree(config, DB, queries, test_queries, res, test_res)

end = time.time()
print("Model Training took {:.2f}s".format(end-start))

save_model(config, my_model)
test_model(config, test_queries, test_res)



