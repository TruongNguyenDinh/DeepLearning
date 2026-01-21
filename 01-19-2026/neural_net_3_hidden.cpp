#include<vector>
#include"active_function.cpp"
#include<iostream>
#define lr  0.01
struct Node{
    double sum;
    double h_export;
    std::vector<double> w;
};
struct Layer{
    std::vector<Node> nodes;
    std::vector<double> bias;
};
double sigma(
    int node,
    const std::vector<double>& b,
    const std::vector<double>& w,
    double x[],
    int num_inp
){   
    double s = b[node];
    for(int i = 1;i<=num_inp;i++){
        s+=w[i]*x[i];
    }
    return s;
}
void init_(Layer& layer_1,Layer& layer_2,Layer& layer_3,Layer &out){
    // Khởi tạo tầng ẩn thứ nhất
    layer_1.nodes.resize(4);
    layer_1.bias = {0.0, 0.3, 0.5, 0.4}; // Tao bias tang 1
    layer_1.nodes[1].w = {0.0, 0.10, 0.20, 0.30};
    layer_1.nodes[2].w = {0.0, 0.40, 0.50, 0.60};
    layer_1.nodes[3].w = {0.0, 0.70, 0.80, 0.90};
    // Khởi tạo tầng ẩn thứ 2
    layer_2.nodes.resize(4); 
    layer_2.bias = {0.0, 0.1, 0.2, 0.6}; // Tao bias tang 2
    layer_2.nodes[1].w ={0.0, 0.20, 0.25, 0.30};
    layer_2.nodes[2].w ={0.0, 0.35, 0.40, 0.45};
    layer_2.nodes[3].w ={0.0, 0.50, 0.55, 0.60};
    // Khởi tạo tầng ẩn thứ 3
    layer_3.nodes.resize(4);
    layer_3.bias = {0.0, 0.5, 0.3, 0.7}; // Tao bias tang 3
    layer_3.nodes[1].w ={0.0, 0.95, 0.90, 0.85};
    layer_3.nodes[2].w ={0.0, 0.80, 0.75, 0.70};
    layer_3.nodes[3].w ={0.0, 0.65, 0.60, 0.55};
    // Out
    out.nodes.resize(4);
    out.nodes[1].w ={0.0, 0.95};
    out.nodes[2].w ={0.0, 0.80};
    out.nodes[3].w ={0.0, 0.65};
}

int main(){
    // Giá trị đầu vào
    double x[] = {0.0, 0.1, 0.2, 0.3};
    //Giá trị mong muốn
    double y[] = {0.0, 1.0 ,2.0, 5.0};
    Layer layer_1;
    Layer layer_2;
    Layer layer_3;
    Layer out;
    init_(layer_1,layer_2,layer_3,out);
    double L = 0;

    for(int epoch = 0; epoch < 10000; epoch ++){
        // =============== Lan truyền xuôi =====================
        // Tính tổng trọng số và đầu ra của tầng 1
        layer_1.nodes[1].sum = sigma(1,layer_1.bias,layer_1.nodes[1].w,x,3);
        layer_1.nodes[1].h_export = ActivationFunction::tanh_act(layer_1.nodes[1].sum);

        layer_1.nodes[2].sum = sigma(2,layer_1.bias,layer_1.nodes[2].w,x,3);
        layer_1.nodes[2].h_export = ActivationFunction::tanh_act(layer_1.nodes[2].sum);

        layer_1.nodes[3].sum = sigma(3,layer_1.bias,layer_1.nodes[3].w,x,3);
        layer_1.nodes[3].h_export = ActivationFunction::tanh_act(layer_1.nodes[3].sum);
        double h1f[] = {
            0,
            layer_1.nodes[1].h_export,
            layer_1.nodes[2].h_export,
            layer_1.nodes[3].h_export
        };

        // Tính tổng trọng số và đầu ra của tầng 2
        layer_2.nodes[1].sum = sigma(1,layer_2.bias,layer_2.nodes[1].w,h1f,3);
        layer_2.nodes[1].h_export = ActivationFunction::ReLU(layer_2.nodes[1].sum);

        layer_2.nodes[2].sum = sigma(2,layer_2.bias,layer_2.nodes[2].w,h1f,3);
        layer_2.nodes[2].h_export = ActivationFunction::ReLU(layer_2.nodes[2].sum);

        layer_2.nodes[3].sum = sigma(3,layer_2.bias,layer_2.nodes[3].w,h1f,3);
        layer_2.nodes[3].h_export = ActivationFunction::ReLU(layer_2.nodes[3].sum);
        double h2f[]= {
            0,
            layer_2.nodes[1].h_export,
            layer_2.nodes[2].h_export,
            layer_2.nodes[3].h_export
        };
        // Tính tổng trọng số và đầu ra của tầng 3
        layer_3.nodes[1].sum = sigma(1,layer_3.bias,layer_3.nodes[1].w,h2f,3);
        layer_3.nodes[1].h_export = ActivationFunction::SELU(layer_3.nodes[1].sum);

        layer_3.nodes[2].sum = sigma(2,layer_3.bias,layer_3.nodes[2].w,h2f,3);
        layer_3.nodes[2].h_export = ActivationFunction::SELU(layer_3.nodes[2].sum);

        layer_3.nodes[3].sum = sigma(3,layer_3.bias,layer_3.nodes[3].w,h2f,3);
        layer_3.nodes[3].h_export = ActivationFunction::SELU(layer_3.nodes[3].sum);
        double h3f[]= {
            0,
            layer_3.nodes[1].h_export,
            layer_3.nodes[2].h_export,
            layer_3.nodes[3].h_export
        };
        // Tính đầu 
        out.nodes[1].h_export = layer_3.nodes[1].h_export*out.nodes[1].w[1] + layer_3.bias[1];
        out.nodes[2].h_export = layer_3.nodes[2].h_export*out.nodes[2].w[1] + layer_3.bias[2];
        out.nodes[3].h_export = layer_3.nodes[3].h_export*out.nodes[3].w[1] + layer_3.bias[3];
        // Tính sai lệch
        L = 0.5*(
            std::pow(out.nodes[1].h_export - y[1],2)+
            std::pow(out.nodes[2].h_export - y[2],2)+
            std::pow(out.nodes[3].h_export - y[3],2)
        );
        if (epoch % 1000 == 0)
            std::cout << "Epoch " << epoch << " Loss = " << L << std::endl;

        if (L < 1e-5) break;
        // =============== Lan truyền ngược =====================
        // Tầng đầu ra
        double delta_out[4];
        for(int i=1;i<=3;i++){
           delta_out[i] = out.nodes[i].h_export-y[i]; 
        }
        // Tầng ẩn số 3
        double delta_3[4];
        for(int i=1;i<=3;i++){
            delta_3[i] =
                delta_out[i]
                * out.nodes[i].w[1]
                * ActivationFunction::dSELU(layer_3.nodes[i].sum);
        }
        // Tẩng ẩn số 2
        double delta_2[4];
        for(int i=1;i<=3;i++){
            double grad = 0.0;
            for(int k=1;k<=3;k++){
                grad += delta_3[k] * layer_3.nodes[k].w[i];
            }
            delta_2[i] =
                grad * ActivationFunction::dReLU(layer_2.nodes[i].sum);
        }
        // Tẩng ẩn đầu tiên
        double delta_1[4];
        for(int i=1;i<=3;i++){
            double grad = 0.0;
            for(int k=1;k<=3;k++){
                grad += delta_2[k] * layer_2.nodes[k].w[i];
            }
            delta_1[i] =
                grad *
                ActivationFunction::dtanh_from_output(layer_1.nodes[i].h_export);
        }
        // =============== Cập nhật trọng số =====================
        // layer 3
        for(int j = 1; j <= 3; j++){
            for(int i = 1; i <= 3; i++){
                layer_3.nodes[j].w[i] -= lr * delta_3[j] * h2f[i];
            }
            layer_3.bias[j] -= lr * delta_3[j];
        }
        // Layer 2
        for(int j = 1; j <= 3; j++){
            for(int i = 1; i <= 3; i++){
                layer_2.nodes[j].w[i] -= lr * delta_2[j] * h1f[i];
            }
            layer_2.bias[j] -= lr * delta_2[j];
        }
        // Layer 1
        for(int j = 1; j <= 3; j++){
            for(int i = 1; i <= 3; i++){
                layer_1.nodes[j].w[i] -= lr * delta_1[j] * x[i];
            }
            layer_1.bias[j] -= lr * delta_1[j];
        }
        // output layer
        for(int i=1;i<=3;i++){
            out.nodes[i].w[1] -= lr * delta_out[i] * h3f[i];
        }
    }
    std::cout << "\nFinal output:\n";
    std::cout << "y1 = " << out.nodes[1].h_export << "\n";
    std::cout << "y2 = " << out.nodes[2].h_export << "\n";
    std::cout << "y2 = " << out.nodes[3].h_export<< "\n";
    std::cout << "Final Loss = " << L << std::endl;   
}