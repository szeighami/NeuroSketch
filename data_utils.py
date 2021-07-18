import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets


def get_nn_res(db, queries, test_queries, k_th, return_dist, no_threads):

    nbrs = NearestNeighbors(n_neighbors=k_th, algorithm='ball_tree').fit(db)
    if no_threads == 1 or db.shape[0] > 20*(10**6):
        if db.shape[0] > 20*(10**6):
            print("Database too large for multi threading. Only using one thread")
        distances, indices = nbrs.kneighbors(queries)
    else:
        import multiprocessing
        pool = multiprocessing.Pool(processes=no_threads)
        outputs = pool.map(nbrs.kneighbors, [queries[i*(queries.shape[0]//no_threads):(i+1)*(queries.shape[0]//no_threads)] for i in range(no_threads)])
        distances = np.concatenate([outputs[i][0] for i in range(no_threads)], axis=0)
        indices = np.concatenate([outputs[i][1] for i in range(no_threads)], axis=0)

    if return_dist: 
        res = distances[:, k_th-1]
    else:
        res = db[indices[:, k_th-1].reshape(-1), :]

    distances, indices = nbrs.kneighbors(test_queries)
    if return_dist:
        test_res = distances[:, k_th-1]
    else:
        test_res = db[indices[:, k_th-1].reshape(-1), :]

    return res, test_res

def get_nn_data(n, dim, return_dist, train_size, test_size, k_th, data_loc, max_val, no_threads):
    db = np.load(data_loc)
    #shuffle in case there is sub-sampling
    np.random.shuffle(db)
    db = db[:n, :]

    queries = db[:train_size, :]
    test_queries = db[-test_size:, :]
    res, test_res = get_nn_res(db, queries, test_queries, k_th, return_dist, no_threads)

    return db, queries, test_queries, res, test_res 

def calc_median_with_angle(x):
    x,  keys, vals, dim, agg_type = x
    getmask = lambda keys, x, d: np.logical_and(np.tan(x[-1])*keys[:, 0]+x[d+2] - np.tan(x[-1])*x[d]<keys[:, 1], (1/np.tan(x[-1]))*keys[:, 0]+x[d+2] - (1/np.tan(x[-1]))*x[d]<keys[:, 1])
    median_w_empty = lambda a: 0 if a.shape[0] == 0 else np.median(a)
    return np.array([median_w_empty(vals[np.logical_and(getmask(keys, x[i], 0), getmask(keys, x[i], 1))]) for i in range(x.shape[0])]).reshape((-1, 1))

def calc_agg_values(x):
    x,  keys, vals, dim, agg_type = x
    std_w_empty = lambda a: 0 if a.shape[0] == 0 else np.std(a)
    sum_w_empty = lambda a: 0 if a.shape[0] == 0 else np.sum(a)
    mean_w_empty = lambda a: 0 if a.shape[0] == 0 else np.mean(a)
    if agg_type == 2:
        return np.array([np.sum([np.logical_and.reduce([np.logical_and(keys[:, d]>=x[i, d, 0], keys[:, d]<x[i, d, 1]) for d in range(dim)])]) for i in range(x.shape[0])]).reshape((-1, 1))
    elif agg_type == 0:
        return np.array([std_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=x[i, d, 0], keys[:, d]<x[i, d, 1]) for d in range(dim)])]) for i in range(x.shape[0])]).reshape((-1, 1))
    elif agg_type == 1:
        return np.array([mean_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=x[i, d, 0], keys[:, d]<x[i, d, 1]) for d in range(dim)])]) for i in range(x.shape[0])]).reshape((-1, 1))
    elif agg_type == 3:
        return np.array([sum_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=x[i, d, 0], keys[:, d]<x[i, d, 1]) for d in range(dim)])]) for i in range(x.shape[0])]).reshape((-1, 1))

def get_range_agg_queries_and_res(db_dim, size, max_val, q_range, predicate_att_vals, measure_att_vals, agg_type, data_size, active_dim_pairs, with_angle, no_threads):
    predicate_att_vals = predicate_att_vals[:data_size]
    measure_att_vals = measure_att_vals[:data_size]

    queries = np.zeros((size, db_dim, 2))
    queries[:, :, 0] = -max_val/2
    queries[:, :, 0] = queries[:, :, 0]
    queries[:, :, 1] = max_val/2                                        
    queries[:, :, 1] = queries[:, :, 1]
    queries = queries.reshape((size, db_dim*2))
    size_per_pred_pair = size//active_dim_pairs.shape[0]

    for i, x in enumerate(active_dim_pairs):
        for curr_dim in x:
            if q_range == 1:
                curr_dim_queries = np.sort((np.random.rand(size_per_pred_pair, 2)-0.5)*max_val, axis=-1)
            else:
                begin_queries =  (np.random.rand(size_per_pred_pair, 1))*(max_val*(1-q_range))+(-1*max_val*0.5)
                end_queries =  begin_queries+q_range*max_val
                curr_dim_queries = np.concatenate([begin_queries, end_queries], axis=-1)
            queries[size_per_pred_pair*i:size_per_pred_pair*(i+1), 2*curr_dim] = curr_dim_queries[:, 0]
            queries[size_per_pred_pair*i:size_per_pred_pair*(i+1), 2*curr_dim+1] = curr_dim_queries[:, 1]

    queries = queries.reshape((size, db_dim, 2))

    if with_angle:
        queries = queries.reshape((size, 2*db_dim))
        angles = np.random.rand(size, 1)*np.pi/2
        queries = np.concatenate([queries, angles], axis=1)
        func = calc_median_with_angle
    else:
        func = calc_agg_values

    if no_threads == 1 or predicate_att_vals.shape[0] > 20*(10**6):
        if predicate_att_vals.shape[0] > 20*(10**6):
            print("Database too large for multi threading. Only using one thread")
        res = func((queries, predicate_att_vals, measure_att_vals, db_dim, agg_type))
    else:
        import multiprocessing
        pool = multiprocessing.Pool(processes=no_threads)
        outputs = pool.map(func, [(queries[i*(queries.shape[0]//no_threads):(i+1)*(queries.shape[0]//no_threads)], predicate_att_vals, measure_att_vals, db_dim, agg_type) for i in range(no_threads)])
        res = np.concatenate([outputs[i] for i in range(no_threads)], axis=0)

    queries = queries.reshape(queries.shape[0], db_dim*2)
    return queries, res


def get_raq_data(n, db_dim, pred_dim, train_size, test_size, q_range, agg_type, max_val, db_path, measure_att,  train_data_size, no_active_dim, no_active_dim_pairs, with_angle, no_threads):

    db = np.load(db_path)
    #shuffle in case there is sub-sampling
    np.random.shuffle(db)

    measure_att_vals = db[:, measure_att] 

    #normalizing inputs to neural neworks for better learning, queries can be mapped back if desired
    predicate_att_vals = np.delete(db, measure_att, axis=1)
    min_vals = np.min(predicate_att_vals, axis=0)
    max_vals = np.max(predicate_att_vals, axis=0)
    predicate_att_vals = ((predicate_att_vals-min_vals)/(max_vals-min_vals)-0.5)*max_val

    norm_db = np.append(predicate_att_vals, np.reshape(measure_att_vals, (-1, 1)), 1)


    if no_active_dim == 1:
        active_dim_pairs = np.array(range(db_dim)).reshape((-1, 1))
    else:
        active_dim_pairs = np.zeros((no_active_dim_pairs, no_active_dim), dtype=int)
        for row in range(no_active_dim_pairs):
            active_dim_pairs[row, :] = np.random.choice(db_dim, size=no_active_dim, replace=False)
    
    if train_data_size == 0:
        train_data_size = n  
    queries, res = get_range_agg_queries_and_res(db_dim, train_size, max_val, q_range, predicate_att_vals, measure_att_vals, agg_type,  train_data_size, active_dim_pairs, with_angle, no_threads)
    test_queries, test_res = get_range_agg_queries_and_res(db_dim, test_size, max_val, q_range, predicate_att_vals, measure_att_vals, agg_type,  n, active_dim_pairs, with_angle, no_threads)
    #removing null values
    mask = (res != 0).reshape(-1)
    queries = queries[mask]
    res = res[mask]
    mask = (test_res != 0).reshape(-1)
    test_queries = test_queries[mask]
    test_res = test_res[mask]

    return norm_db, queries, test_queries, res, test_res






