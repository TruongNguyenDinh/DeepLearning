#include<bits/stdc++.h>
#define af(t) (1.0 / (1.0 + exp(-(t))))
#define H_1(w, b, x1, x2) (w[1]*x1+w[2]*x2 + b[1])
#define H_2(w, b, x1, x2) (w[3]*x1+w[4]*x2 + b[1])
#define y_1(w, b, h1f, h2f) (w[5]*h1f+w[6]*h2f + b[2])
#define y_2(w, b, h1f, h2f) (w[7]*h1f+w[8]*h2f + b[2])
#define Lf(y1f,y2f,x1_d,x2_d) (0.5*(pow(y1f-x1_d,2)+ pow(y2f-x2_d,2)))
int main(){
    // Input
    double x1 = 0.05;
    double x2 = 0.1;
    // Output - desire
    double x1_d = 0.01;
    double x2_d = 0.99;
    // Weight
    std::vector<double> w = {0,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50};
    std::vector<double> b = {0,0.4, 0.45}; // Khai báo b
    // Chạy
    double lr = 0.2;
    double y1 = 0, y2 = 0, L = 0;

    for (int epoch = 0; epoch < 1; epoch++) {

        // ===== Truyền xuôi  =====
        double h1 = H_1(w,b,x1,x2);
        double h2 = H_2(w,b,x1,x2);

        double h1f = af(h1);
        double h2f = af(h2);

        y1 = af(y_1(w,b,h1f,h2f));
        y2 = af(y_2(w,b,h1f,h2f));

        L = Lf(y1,y2,x1_d,x2_d);

        if (epoch % 1000 == 0)
            std::cout << "Epoch " << epoch << " Loss = " << L << std::endl;

        if (L < 1e-5) break;

        // ===== Lan truyền ngược  =====
        double dy1 = (y1 - x1_d) * y1 * (1 - y1);
        double dy2 = (y2 - x2_d) * y2 * (1 - y2);

        double dh1 = (dy1*w[5] + dy2*w[7]) * h1f * (1 - h1f);
        double dh2 = (dy1*w[6] + dy2*w[8]) * h2f * (1 - h2f);
    
        //
        double L1 = dh1*x1;
        double L2 = dh1*x2;
        double L3 = dh2*x1;
        double L4 = dh2*x2;
        double L5 = dy1*h1f;
        double L6 = dy1*h2f;
        double L7 = dy2*h1f;
        double L8 = dy2*h2f;
        std::cout<<L1<<std::endl<<L2<<std::endl<<L3<<std::endl<<L4<<std::endl<<L5<<std::endl<<L6<<std::endl<<L7<<std::endl<<L8;

        // Cập nhật trọng số cho đầu ra sau hidden layer
        w[5] -= lr * dy1 * h1f;
        w[6] -= lr * dy1 * h2f;
        w[7] -= lr * dy2 * h1f;
        w[8] -= lr * dy2 * h2f;
        b[2] -= lr * (dy1 + dy2);

        // Cập nhật trọng số cho các đầu vào hidden layer
        w[1] -= lr * dh1 * x1;
        w[2] -= lr * dh1 * x2;
        w[3] -= lr * dh2 * x1;
        w[4] -= lr * dh2 * x2;
        b[1] -= lr * (dh1 + dh2);
    }

    std::cout << "\nFinal output:\n";
    std::cout << "y1 = " << y1 << "\n";
    std::cout << "y2 = " << y2 << "\n";
    std::cout << "Final Loss = " << L << std::endl;    
}