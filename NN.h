#include <cstring>
#include <math.h> 
#include<string>
#include <fstream>

void relu(float* x, int dim, float* res) 
{
    for (int i = 0; i < dim; i++) 
    {
        if (x[i] >= 0)
            res[i] = x[i];
        else
            res[i] = 0;
    }
}


//W:dim1xdim2, x:dim1, res:dim2
void multiply(float** w, float* x, int dim1, int dim2, float* res) 
{ 
    int i, j; 
    for (i = 0; i < dim2; i++) 
    { 
        res[i] = 0;
        for (j = 0; j < dim1; j++) 
        { 
            res[i]+= w[j][i] * x[j]; 
        } 
    } 
} 

void elem_add(float* b, float* x, int dim, float* res) 
{ 
    int i; 
    for (i = 0; i < dim; i++) 
    { 
        res[i] = b[i]+x[i];
    } 
} 

void init_layer(std::ifstream* file, std::ifstream* file_d, int in_size, int out_size, float**& W, float*& b)
{
    std::string line;
    //std::getline(*file, line);

    W = new float*[in_size];
    for (int i =0; i < in_size; i++)
    {
        W[i] = new float[out_size];
        for (int j = 0; j < out_size; j++)
        {
            file_d->read((char*)&W[i][j], sizeof(float));
            //int next_del = line.find(',');
            //std::string vals = line.substr(0, sizeof(float));
            //std::memcpy(&W[i][j], vals.c_str(), sizeof(float));

            //line = line.substr(sizeof(float)+1, line.length());
        }
    }

    b = new float[out_size];
    std::getline(*file, line);
    //std::getline(*file, line);
    for (int i =0; i < out_size; i++)
    {
        file_d->read((char*)&b[i], sizeof(float));
        //std::string vals = line.substr(0, sizeof(float));
        //std::memcpy(&b[i], vals.c_str(), sizeof(float));

        //line = line.substr(sizeof(float)+1, line.length());
        //int next_del = line.find(',');
        //std::string vals = line.substr(0, next_del);
        //b[i] = std::stod(vals);

        //line = line.substr(next_del+1, line.length());
    }
}

class NN
{
public:
    NN(std::string path)
    {
        this->path = path;
        std::ifstream file(path);
        std::ifstream file_d(path+"d");

        std::string line;

        std::getline(file, line);
        no_layers = std::stoi(line);
        

        sizes = new int[no_layers+1];
        Ws = new float**[no_layers];
        bs = new float*[no_layers];

        for (int i = 0; i < no_layers; i++)
        {

            std::getline(file, line);
            if (i == 0)
            {
                sizes[i] = std::stoi(line.substr(0, line.find(',')));
            }
            sizes[i+1] = std::stoi(line.substr(line.find(',')+1, line.length()));


            init_layer(&file, &file_d, sizes[i], sizes[i+1], Ws[i], bs[i]);
        }
    }

    void call(float* x, float* res)
    {
        float* in = x;
        for (int i = 0; i < no_layers; i++)
        {
            float* res1 = new float[sizes[i+1]];
            float* res2 = new float[sizes[i+1]];
            float* res3 = new float[sizes[i+1]];

            float* res0;
            res0 = in;

            multiply(Ws[i], res0, sizes[i], sizes[i+1], res1);
            if (i != 0)
                delete[] res0;

            elem_add(bs[i], res1, sizes[i+1], res2);

            delete[] res1;
            if (i == 0)
            {
                relu(res2, sizes[i+1], res3);
                delete[] res2;
            }
            else if (i < no_layers - 1)
            {
                relu(res2, sizes[i+1], res3);
                delete[] res2;
            }
            else
            {
                delete[] res3;
                res3 = res2;
            }

            in = res3;
        }

        for (int i = 0; i < sizes[no_layers]; i++)
            res[i] = in[i];
        delete[] in;
    }

    void print_nn()
    {
        for (int z =0; z < no_layers; z++)
        {
            std::cout << "W" << z+1 << " (" << sizes[z] << "," << sizes[z+1] << "): "; 
            for (int i = 0; i < sizes[z]; i++)
            {
                for (int j = 0; j < sizes[z+1]; j++)
                    std::cout << Ws[z][i][j] << ',';
            }
            std::cout <<'\n';

            std::cout << "b" << z+1 << " (" << sizes[z+1] << "): ";
            for (int i = 0; i < sizes[z+1]; i++)
            {
                std::cout << bs[z][i] << ',';
            }
            std::cout <<'\n';
        }

    }

    std::string get_path()
    {
        return path;
    }
    int get_input_dim()
    {
        return sizes[0];
    }

    int get_size()
    {
        int total_size = 0;
        for (int i = 0; i < no_layers; i++)
            total_size += sizes[i]*sizes[i+1]+sizes[i+1];
        return total_size;
    }

private:
    std::string path;

    int no_layers;
    int* sizes;
    float*** Ws;
    float** bs;
};


