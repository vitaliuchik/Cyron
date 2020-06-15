#include "snn.h"


void snn_tutorial() {
    auto X = Data::read_data("../data/mnist_x.txt");
    auto Y = Data::read_data("../data/mnist_y.txt");

    X = Data::scale(X);
    auto [X_train, X_test] = Data::split(X, 0.2);
    auto [Y_train, Y_test] = Data::split(Y, 0.2);

    // Y preparation
    Data::set Y_train_prepared;
    Data::vector temp;
    for (const auto& v: Y_train) {
        for (int i = 0; i < 10; i++) {
            temp.push_back(0);
        }
        temp[(int) v[0]] = 1;
        Y_train_prepared.push_back(temp);
        temp.clear();
    }

    // training
    std::vector<int> layer_dimensions = {64, 32, 10};
    auto model = StandardNeuralNetwork(X_train, Y_train_prepared, layer_dimensions);


    auto start_time = get_current_time_fenced();
    model.train(1000, 0.25);
    auto finish_time = get_current_time_fenced();
    auto Y_pred = model.predict(X_test);


    // transforming Y
    Data::set Y_pred_transformed;
    for (const auto& v : Y_pred) {
        double max_val = v[0];
        int max_ind = 0;
        for (int j = 1; j < v.size(); j++) {
            if (v[j] > max_val) {
                max_val = v[j];
                max_ind = j;
            }
        }
        temp.push_back(max_ind);
        Y_pred_transformed.push_back(temp);
        temp.clear();
    }

    Data::printData(Y_pred_transformed);
    std::cout << std::endl;
    Data::printData(Y_test);
    std::cout << "Score: " << Data::score(Y_pred_transformed, Y_test) << std::endl;
//    std::cout << "Duration: " << to_us(finish_time - start_time) << std::endl;

}
