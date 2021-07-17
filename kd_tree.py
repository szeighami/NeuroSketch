import subprocess
import numpy as np

class Node:
    def __init__(self, dim=None, split_dim=None, fanout=None, x=None):
        self.dim = dim
        self.split_dim = split_dim
        self.fanout = fanout
        self.children = None

        if fanout is None:
            self.split_points = None
        else:
            self.split_points = np.zeros((fanout-1,))

        if x is not None:
            self.set_split_point(x)


    def set_split_point(self, x):
        indx = np.argsort(x[:, self.split_dim])
        size = x.shape[0]

        num_in_lower = np.sum(x[:, self.split_dim] < x[indx[size//self.fanout], self.split_dim])
        num_in_upper = x.shape[0]-num_in_lower
        if min(num_in_lower, num_in_upper) < 0.2*x.shape[0]:
            self.split_dim=(self.split_dim+1)%x.shape[1]
            self.set_split_point(x)
            return 

        for i in range(self.fanout-1):
            self.split_points[i] = x[indx[(i+1)*size//self.fanout], self.split_dim]

    def get_params(self, file_name, node_id):
        if self.children == None:
            return ""

        str_cnt = str(self.split_dim)+':'
        for i in range(self.fanout-1):
            str_cnt += str(self.split_points[i]) + ','
        str_cnt+='\n'

        for i in range(self.fanout):
            str_cnt+=self.children[i].get_params(file_name, node_id+str(i+1))
        return str_cnt
        
def train_model(x, y, test_x, test_y, processes, no_processes, no, base_name, path, path_to_neurodb):
    leaf = Node()

    if len(processes) == no_processes:
        for p in processes:
            p.wait()
        processes=[]

    np.save('queries'+str(no)+'.npy', x);
    np.save('res'+str(no)+'.npy', y);
    np.save('test'+str(no)+'_queries.npy', test_x);
    np.save('test'+str(no)+'_res.npy', test_y);

    print("training model no "+str(no)+" --------------")
    p = subprocess.Popen(["python", path_to_neurodb+"/fit_base.py", str(no), base_name, path])  
    processes.append(p)

    return leaf, processes

def get_child_res(i, x, root, fanout, test_x, y, test_y):
    if i == 0:
        less = x[:, root.split_dim] < root.split_points[i]
        indx =less 
        test_less = test_x[:, root.split_dim] < root.split_points[i]
        test_indx =test_less 
    elif i == fanout-1:
        more = x[:, root.split_dim] >= root.split_points[i-1]
        indx =more 
        test_more = test_x[:, root.split_dim] >= root.split_points[i-1]
        test_indx =test_more 
    else:
        less = x[:, root.split_dim] < root.split_points[i]
        more = x[:, root.split_dim] >= root.split_points[i-1]
        indx = less and more
        test_less = test_x[:, root.split_dim] < root.split_points[i]
        test_more = test_x[:, root.split_dim] >= root.split_points[i-1]
        test_indx = test_less and test_more

    x_i=x[indx, :]
    test_x_i=test_x[test_indx, :]
    y_i=y[indx, :]
    test_y_i=test_y[test_indx, :]
    return x_i, test_x_i, y_i, test_y_i

def build_tree(depth, fanout, dim, x, y, test_x, test_y, processes, base_name, path, no, split_dim, no_processes, path_to_neurodb):
    if depth == 0:
        leaf, processes = train_model(x, y, test_x, test_y, processes, no_processes, no, base_name, path, path_to_neurodb)
        no+=1
        return leaf, no, processes

    root = Node(dim, split_dim, fanout, x)

    root.children = []
    for i in range(fanout):
        x_i, test_x_i, y_i, test_y_i = get_child_res(i, x, root, fanout, test_x, y, test_y) 
        child, no, processes = build_tree(depth-1, fanout, dim, x_i, y_i, test_x_i, test_y_i, processes, base_name, path+str(i+1), no, (root.split_dim+1)%dim, no_processes, path_to_neurodb)
        root.children.append(child)

    return  root, no, processes


