#ifndef CYRON_EXPLORATION_H
#define CYRON_EXPLORATION_H

#include <vector>
#include <tuple>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Data {
public:
    typedef std::vector<double> vector;
    typedef std::vector<vector> set;

    static std::tuple<set, set> split(const set& data, double test_size);

    static set read_data(const std::string& path);

    static set scale(const set& data);

    static double mean(const vector& v);

    static double sd(const vector& v);

    static double score(const set& Y_pred, const set& Y_test);

    static void printData(const set& data);
};

#endif //CYRON_EXPLORATION_H
