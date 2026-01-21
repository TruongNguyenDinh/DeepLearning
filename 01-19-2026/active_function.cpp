#include <cmath>

class ActivationFunction {
public:
    // Activation functions
    static double tanh_act(double x) {
        return std::tanh(x);
    }

    static double ReLU(double x) {
        return x > 0 ? x : 0;
    }

    static double SELU(double x) {
        const double alpha  = 1.67326;
        const double lambda = 1.0507;
        return x > 0 ? lambda * x : lambda * alpha * (std::exp(x) - 1);
    }

    // Derivatives
    static double dtanh_from_output(double y) {
        return 1.0 - y * y;
    }

    static double dReLU(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    static double dSELU(double x) {
        const double alpha  = 1.67326;
        const double lambda = 1.0507;
        return x > 0 ? lambda : lambda * alpha * std::exp(x);
    }
};
