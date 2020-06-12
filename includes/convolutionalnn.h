#ifndef CYRON_CONVOLUTIONALNN_H
#define CYRON_CONVOLUTIONALNN_H

#include "exploration.h"

#include <map>
#include <mutex>
#include <random>

#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <eigen3/Eigen/Dense>


class ConvolutionalNeuralNetwork {
    typedef Eigen::MatrixXd matrix;
    typedef Eigen::VectorXd vector;
    typedef std::vector<vector> set;
    typedef std::unordered_map<std::string, matrix> matrix_map;
public:
    ConvolutionalNeuralNetwork(const Data::set& X_train, const Data::set& Y_train, const std::vector<int>& layer_dimensions);

    void train(int iter_num = 1000, double learning_rate = 0.25, int num_threads = 1);

    Data::set predict(const Data::set& X_test);
private:

    matrix_map parameters;
    std::vector<int> layer_dims;
    int layer_num;
    set X, Y;

    // initialization
    void initialize_parameters() {
        std::random_device r;
        std::default_random_engine eng{r()};
        std::uniform_real_distribution<double> dist(0, 1);

        for (int i = 1; i < layer_num; i++) {
            parameters["W" + std::to_string(i)] = Eigen::MatrixXd::NullaryExpr(layer_dims[i],layer_dims[i - 1], [&](){return dist(eng);});
//            parameters["W" + std::to_string(i)] = matrix::Random(layer_dims[i], layer_dims[i - 1]);
            parameters["b" + std::to_string(i)] = vector::Constant(layer_dims[i], 0);
        }
    }

    // activation functions
    vector sigmoid(const vector& Z) {
        return (1.00 / (1 + Eigen::exp(-Z.array()))).matrix();
    }
    vector backward_sigmoid(const vector& Z) {
        auto sigm = sigmoid(Z).array();
        return (sigm * (1 - sigm)).matrix();
    }

    vector relu(const vector& Z) {
        return Z.array().cwiseMax(0).matrix();
    }
    vector backward_relu(const vector& Z) {
        return ((Z.array() > 0).select(vector::Constant(Z.size(), 1).array(), 0)).matrix();
    }

    vector tanh(const vector& Z) {
        return Z.array().tanh().matrix();
    }
    vector backward_tanh(const vector& Z) {
        auto tanh_val = tanh(Z).array();
        return (1 - tanh_val * tanh_val).matrix();
    }

    // propagation
    matrix_map forward_propagation(const vector& X_v, matrix_map params) {
        matrix_map cache;
        cache["A0"] = X_v;
        for (int i = 1; i < layer_num; ++i) {
            cache["Z" + std::to_string(i)] = params["W" + std::to_string(i)] * cache["A" + std::to_string(i-1)] + params["b" + std::to_string(i)];
            cache["A" + std::to_string(i)] = sigmoid(cache["Z" + std::to_string(i)]);
        }
        return cache;
    }

    matrix_map backward_propagation(const vector& Y_v, matrix_map params, matrix_map cache) {
        matrix_map delta_params;
        matrix_map delta_cache;
        delta_cache["dZ" + std::to_string(layer_num - 1)] = cache["A" + std::to_string(layer_num - 1)] - Y_v;
        delta_params["dW" + std::to_string(layer_num - 1)] = delta_cache["dZ" + std::to_string(layer_num - 1)] * cache["A" + std::to_string(layer_num - 2)].transpose();
        delta_params["db" + std::to_string(layer_num - 1)] = delta_cache["dZ" + std::to_string(layer_num - 1)];
        for (int i = layer_num - 2; i > 0; i--) {
            delta_cache["dZ" + std::to_string(i)] = ((params["W" + std::to_string(i+1)].transpose() * delta_cache["dZ" + std::to_string(i+1)]).array() * backward_sigmoid(cache["Z" + std::to_string(i)]).array()).matrix();
            delta_params["dW" + std::to_string(i)] = delta_cache["dZ" + std::to_string(i)] * cache["A" + std::to_string(i-1)].transpose();
            delta_params["db" + std::to_string(i)] = delta_cache["dZ" + std::to_string(i)];
        }
        return delta_params;
    }

    void oneCicle(int i, double learning_rate, std::mutex& mtx) {
        auto cache = forward_propagation(X[i], parameters);
        auto delta_parameters = backward_propagation(Y[i], parameters, cache);
        // update parameters
        mtx.lock();
        for (int layer = 1; layer < layer_num; layer++) {
            parameters["W" + std::to_string(layer)] -= learning_rate * delta_parameters["dW" + std::to_string(layer)];
            parameters["b" + std::to_string(layer)] -= learning_rate * delta_parameters["db" + std::to_string(layer)];
        }
        mtx.unlock();
    }
};

#endif //CYRON_CONVOLUTIONALNN_H
