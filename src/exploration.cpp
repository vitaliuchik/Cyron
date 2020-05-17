#include "../includes/exploration.h"


std::tuple<Data::set, Data::set> Data::split(const Data::set& data, double test_size) {
    if (test_size > 1 || test_size < 0)
        test_size = 0.2;
    int limit = floor(data.size() * (1 - test_size));

    Data::set train;
    for (int i = 0; i < limit; i++)
        train.push_back(data[i]);

    Data::set test;
    for (int i = limit; i < data.size(); i++)
        test.push_back(data[i]);

    return {train, test};
}


Data::set Data::read_data(const std::string& path) {
    Data::set result;
    std::ifstream file(path);

    if(!file){
        std::cerr << "Error opening input file" << std::endl;
        exit(1);
    }
    double x;
    std::string line;
    while( getline(file, line) ) {
        Data::vector v;
        std::istringstream iss(line);
        for (double s; iss >> s;)
            v.push_back(s);
        result.emplace_back(std::move(v));
    }
    file.close();

    return result;
}


double Data::mean(const vector& v)  {
    double sum = 0;
    for (const auto& val: v)
        sum += val;
    return sum / v.size();
}


double Data::sd(const vector& v)  {
    double sum = 0;
    double m = Data::mean(v);
    for (const auto& val: v)
        sum += (val - m) * (val - m);
    return sum / v.size();
}


Data::set Data::scale(const Data::set& data) {
    Data::set result;
    Data::vector temp;
    for (const auto& v: data) {
        double m = Data::mean(v);
        double s = Data::sd(v);
        for (const auto& val: v)
            temp.push_back( (val - m) / s );
        result.push_back(temp);
        temp.clear();
    }

    return result;
}


double Data::score(const Data::set& Y_pred, const Data::set& Y_test) {
    int length = Y_pred.size();
    double coincide = 0;
    for (int i = 0; i < length; i++) {
        if (Y_pred[i] == Y_test[i])
            coincide++;
    }
    return coincide / length;
}


void Data::printData(const Data::set& data) {
    for (const auto& v: data) {
        std::cout << v[0] << "|";
    }
    std::cout << std::endl;
}