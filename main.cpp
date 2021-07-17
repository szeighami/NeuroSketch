#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <chrono> 
#include <string>
#include "NN.h"

struct Node
{
    int dim;
    float* split_points;
    Node** children;
    NN* nn;
};

int fanout;
int dim;
int out_dim;
int depth;

Node* get_new_node()
{
    Node* node = new Node;
    node->nn = NULL;
    node->children = NULL;
    return node;
}

int get_kd_tree_size(Node* root)
{
    if (root->children == NULL)
        return 4*sizeof(int) + root->nn->get_size();

    int total_size = 4*sizeof(int)+fanout*(sizeof(float)+sizeof(int));
    for (int i = 0; i < fanout; i++)
        total_size += get_kd_tree_size(root->children[i]);
    return total_size;
}


void build_kd_tree(Node* root_node, std::ifstream* file, int curr_depth, std::string curr_path)
{
    if (fanout == 1 || curr_depth == depth)
    {
        root_node->nn = new NN(curr_path+".m");
        return;
    }

    std::string line;
    std::getline(*file, line);
    int split_dim = std::stoi(line.substr(0, line.find(':')));
    line = line.substr(line.find(':')+1, line.length());
    
    root_node->split_points = new float[fanout-1];
    root_node->dim = split_dim;
    root_node->children = new Node*[fanout];
    for (int i = 0; i < fanout; i++)
    {
        if (i != fanout-1)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            root_node->split_points[i] =  std::stof(vals);

            line = line.substr(next_del+1, line.length());
        }

        root_node->children[i] = get_new_node();
        std::string path = curr_path;
        path.append(std::to_string(i+1));
        build_kd_tree(root_node->children[i], file, curr_depth+1, path);
    }

}

float dist(float* x, float* y, int input_dim)
{
    float sum = 0;
    for (int i = 0; i < input_dim; i++)
    {
        if (y == 0)
            sum += (x[i])*(x[i]);
        else
            sum += (x[i]-y[i])*(x[i]-y[i]);
    }

    return sqrt(sum);
}

void call_kd_tree(Node* root, float* x, float* res)
{
    if (root->children == NULL)
    {
        root->nn->call(x, res);
        return;
    }

    for (int i = 0; i < fanout-1; i++)
    {
        if (x[root->dim] < root->split_points[i])
            return call_kd_tree(root->children[i], x, res);
    }
    call_kd_tree(root->children[fanout-1], x, res);

}

void print_tree(Node* root, bool with_nn)
{
    if (root->children == NULL)
    {
        if (with_nn)
            root->nn->print_nn();
        return;
    }
    for (int i = 0; i < fanout-1; i++)
    {
        std::cout << root->split_points[i] << '\t';
        std::cout << ";;";
    }
    std::cout << root->dim << std::endl;

    for (int i = 0; i < fanout; i++)
    {
        print_tree(root->children[i], with_nn);
    }
}

void read_queries(float** x, float **y, int& test_size, int out_dim, int dim, std::string query_file)
{
    std::ifstream file_queries(query_file+"_queries.txt");
    std::ifstream file_res(query_file+"_res.txt");

    for (int i = 0; i < test_size; i++)
    {
        x[i] = new float[dim];
        std::string line;
        if (!std::getline(file_queries, line))
        {
            test_size = i;
            break;
        }
        for (int j = 0; j < dim; j++)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            x[i][j] =  std::stof(vals);

            line = line.substr(next_del+1, line.length());
        }

        y[i] = new float[out_dim];
        std::getline(file_res, line);
        for (int j = 0; j < out_dim; j++)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            y[i][j] =  std::stof(vals);

            line = line.substr(next_del+1, line.length());
        }
    }
}

int main(int argc, char** argv)
{
    std::cout << std::setprecision(10);

    fanout = atoi(argv[1]);
    dim = atoi(argv[2]);
    out_dim = atoi(argv[3]);
    depth = atoi(argv[4]);
        
    std::string query_file = argv[5];
    std::string model_file = argv[6];
    int test_size = atoi(argv[7]);
    bool output_result = atoi(argv[8]) == 1;

    std::ofstream file_output;
    if (output_result)
        file_output.open(query_file+"_out.txt");

    float** x = new float*[test_size];
    float** y = new float*[test_size];

    read_queries(x, y, test_size, out_dim, dim, query_file);

    std::ifstream infile(model_file+"_tree.m");

    Node* root_node = get_new_node();
    build_kd_tree(root_node, &infile, 0, model_file);


    auto start = std::chrono::high_resolution_clock::now(); 

    float mse = 0;
    float mean_rel_acc = 0;
    float mean_norm = 0;
    for (int i = 0; i < test_size; i++)
    {
        float* res = new float[out_dim];
        call_kd_tree(root_node, x[i], res);

        mean_rel_acc += dist(res, y[i], out_dim)/dist(y[i], 0, out_dim);
        mean_norm += dist(y[i], 0, out_dim);
        mse += dist(res, y[i], out_dim);

        if (output_result)
        {
            for (int z = 0; z < out_dim; z++)
                file_output <<  res[z] << ",";
            file_output <<  std::endl;
        }

    }
    mse = mse/test_size;
    mean_norm = mean_norm/test_size;
    mean_rel_acc = mean_rel_acc/test_size;

    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 

    std::cout << "time:" << ((float)duration.count())/((float)test_size) << " micro seconds" <<std::endl; 
    std::cout << "rmse:" << mse << std::endl; 
    std::cout << "avg rel acc:" << mean_rel_acc << std::endl; 
    std::cout << "normalized rmse:" << mse/mean_norm << std::endl; 
    std::cout << "mean result norm:" << mean_norm << std::endl; 
    std::cout << "model size:" << get_kd_tree_size(root_node)*4/(1024.0) << "KB" << std::endl; 

    return 0;
}
