#include "dynet/training.h"
#include "dynet/expr.h"
#include "RNN.hpp"
#include "Noise.hpp"

namespace dy = dynet;
using namespace dy;

#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;


std::tuple<unsigned, unsigned, unsigned> read_args(int argc, char* argv[]){
    if(argc < 4) {
        std::cout << "Arguments expected for: \n";
        std::cout << " * network size \n";
        std::cout << " * the max length of noise \n";
        std::cout << " * number of training itterations \n";

        throw std::invalid_argument("Incorrect number of arguments.");
    }

    unsigned size = std::atoi(argv[1]);
    unsigned noise_len = std::atoi(argv[2]);
    unsigned epoch_max = std::atoi(argv[3]);

    std::cout << " * number of neurons in recurrent layer : " <<  size << "\n";
    std::cout << " * the max length of noise              : " << noise_len << "\n";
    std::cout << " * number of training itterations       : " << epoch_max <<  "\n";

    return std::make_tuple(size, noise_len, epoch_max);
}

vector<dy::real> train(
        const vector<dy::real> X, 
        const vector<dy::real> Y, 
        const float rate, 
        const unsigned size, 
        const unsigned noise_len, 
        const unsigned epoch_max
){
    ComputationGraph cg;
    RNN net(size);
    SimpleSGDTrainer trainer(net.params, rate);

    Noise rng(-2.0, 2.0);

    vector<dy::real> out(epoch_max);
    unsigned batch = (unsigned) Y.size();

    for(int epoch = 0; epoch < epoch_max; epoch ++){
        cg.clear();

        auto X_ = rng.create_data(X, noise_len);
        auto x = input(cg, Dim({noise_len + 1}, batch), &X_); 
        auto y = input(cg, Dim({1}, batch), &Y);

        auto y_pred = net.forward(cg, x);

        y_pred = reshape(y_pred, Dim({batch}, 1));
        y      = reshape(y,      Dim({batch}, 1));
        auto loss = binary_log_loss(y_pred, y);

        auto my_loss = as_scalar(cg.forward(loss));

        cg.backward(loss);
        trainer.update();

        out[epoch] = my_loss;
    }

    //cout << "E = " << out[0] << " " << out[epoch_max - 1] << endl;
    return(out);
}


int main(int argc, char* argv[]){
    //int argc; char** argv;
    dy::initialize(argc, argv);

    auto [size, noise_len, epoch_max] = read_args(argc, argv);
    //unsigned int size, noise_len, epoch_max; tie(size, noise_len, epoch_max) = make_tuple(10, 5 , 500);

    const vector<dy::real> X = {-1.0, 1.0};
    const vector<dy::real> Y = {0.0, 1.0};

    unsigned success;
    for(float rate = 3.55; rate > 0.0; rate -= 0.25){
        stringstream file_path; 
        ofstream error_log;

        // rates, size, activation, 
        file_path << "./out/" << noise_len << '_' << size << '_' << rate << ".csv"; 
        error_log.open(file_path.str());

        success = 0;
        for(unsigned i = 0; i < 100; i++){
            auto errs = train(X, Y, rate, size, noise_len, epoch_max);

            if(errs[epoch_max - 1] < 0.18) success++;

            for(auto e: errs)
                error_log << e << ',';

            error_log << endl;
        }

        cout << "rate: " << rate << " success: " << success << "%" << endl;

        error_log.close();
    }
}
