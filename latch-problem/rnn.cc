#include "dynet/training.h"
#include "dynet/expr.h"
#include "RNN.hpp"
#include "Noise.hpp"

namespace dy = dynet;
using namespace dy;

#include <tuple>
#include <iostream>
#include <stdexcept>

using namespace std;


std::tuple<unsigned, unsigned, unsigned> read_args(int argc, char* argv[]){
    if(argc < 4) {
        std::cout << "Arguments expected for: \n";
        std::cout << " * number of neurons in recurrent layer \n";
        std::cout << " * the max length of noise \n";
        std::cout << " * number of training itterations \n";

        throw std::invalid_argument("Incorrect number of arguments.");
    }

    unsigned neuron_count = std::atoi(argv[1]);
    unsigned L_max = std::atoi(argv[2]);
    unsigned epoch_max = std::atoi(argv[3]);

    std::cout << " * number of neurons in recurrent layer : " <<  neuron_count << "\n";
    std::cout << " * the max length of noise              : " << L_max << "\n";
    std::cout << " * number of training itterations       : " << epoch_max <<  "\n";

    return std::make_tuple(neuron_count, L_max, epoch_max);
}

void train(
        const vector<dy::real> X, 
        const vector<dy::real> Y, 
        const unsigned neuron_count, 
        const unsigned L, 
        const unsigned epoch_max)
{
    ComputationGraph cg;
    RNN net(neuron_count);
    SimpleSGDTrainer trainer(net.params, 0.05);

    Noise rng(-1.0, 1.0);

    vector<dy::real> out(epoch_max);
    unsigned batch = (unsigned) Y.size();

    for(int epoch = 0; epoch < epoch_max; epoch ++){
        cg.clear();

        auto X_ = rng.create_data(X, L);
        auto x = input(cg, Dim({L + 1}, batch), &X_); 
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

    cout << "L = " << L << " - ";
    cout << "E = " << out[0] << " " << out[epoch_max - 1] << endl;
}


int main(int argc, char* argv[]){
    //int argc; char** argv;
    dy::initialize(argc, argv);

    auto [neuron_count, L_max, epoch_max] = read_args(argc, argv);
    //unsigned int neuron_count, L, epoch_max; tie(neuron_count, L, epoch_max) = make_tuple(10, 5 , 500);

    const vector<dy::real> X = {-1.0, 1.0};
    const vector<dy::real> Y = {0.0, 1.0};
    for(unsigned L = 1; L <= L_max; L++){
        train(X, Y, neuron_count, L, epoch_max);
    }
}
