#include "../includes/standardnn.h"


StandardNeuralNetwork::StandardNeuralNetwork(const Data::set& X_train, const Data::set& Y_train, const std::vector<int>& layer_dimensions) {

    assert(X_train.size() == Y_train.size());
    StandardNeuralNetwork::vector X_temp = StandardNeuralNetwork::vector(X_train[0].size());
    StandardNeuralNetwork::vector Y_temp = StandardNeuralNetwork::vector(Y_train[0].size());

    for (int i = 0; i < X_train.size(); i++) {
        for (int j = 0; j < X_train[0].size(); j++) {
            X_temp[j] = X_train[i][j];
        }
        for (int j = 0; j < Y_train[0].size(); j++) {
            Y_temp[j] = Y_train[i][j];
        }
        this->X.push_back(X_temp);
        this->Y.push_back(Y_temp);
    }
    this->layer_dims = layer_dimensions;
    this->layer_num = layer_dimensions.size();

}


void StandardNeuralNetwork::train(int iter_num, double learning_rate) {
    initialize_parameters();
    std::mutex mtx;
//    if (num_threads > 0)
//        tbb::task_scheduler_init init(num_threads);

    for (int iter = 0; iter < iter_num; iter++) {
//        if (iter % 100 == 0) std::cout << "Iteraion: " << iter << std::endl;

        tbb::parallel_for( tbb::blocked_range<int>(0,X.size()),
                           [&](tbb::blocked_range<int> r)
        {
            for (int i=r.begin(); i<r.end(); ++i) {
                oneCicle(i, learning_rate, mtx);
            }
        });
    }
}


Data::set StandardNeuralNetwork::predict(const Data::set& X_test) {

    set X_res;
    vector X_temp = vector(X_test[0].size());
    for (int i = 0; i < X_test.size(); i++) {
        if (X_test[i].size() != X_test[0].size()) std::cout << X_test[i].size() << std::endl;
        for (int j = 0; j < X_test[0].size(); j++) {
            X_temp[j] = X_test[i][j];
        }
        X_res.push_back(X_temp);
    }

    Data::set Y_res;
    Data::vector Y_temp;
    for (const auto& X_re : X_res) {
        auto cache = forward_propagation(X_re, parameters);
        for (int i = 0; i < layer_dims[layer_num - 1]; i++) {
            Y_temp.push_back(cache["A" + std::to_string(layer_num-1)](i));
        }
        Y_res.push_back(Y_temp);
        Y_temp.clear();
    }

    return Y_res;
}