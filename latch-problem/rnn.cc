#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"

namespace dy = dynet;

#include <tuple>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace std;

class RNN {
    private:
        ParameterCollection m;
        Parameter hidden;
        Parameter output;
        unsigned int state_size;
    public: 
        ComputationGraph cg;

        RNN(neuron_count){
            state_size = (unsigned int) neuron_count;
            hidden = m.add_parameters({state_size, 1 + state_size})
            output = m.add_parameters({1, state_size})
        }

        vector<dy::real> forward(vector<dy::real> X[], unsigned int L){
            int batch_size = X.size() / L ;

            Expression x = input(cg, Dim({L}, batch_size), &X); 
            Expression h = dy::realzeroes(cg, Dim({state_size}, batch_size));

            Expression w;
            h = concatenate({x, h});
            w = parameter(cg, hidden);


            //h = state or self.empty_state
            //for x in data.split(1, dim=1):
            //    x, h = self.forward_state(x, h)

            //#progress with empty input for single output
            //# x, h = self.forward_state(self.empty_input, h)
            //return x, h
        }
}

std::tuple<int, int, int> read_args(int argc, char* argv[]){
    if(argc < 4) {
        std::cout << "Arguments expected for: \n";
        std::cout << " * number of neurons in recurrent layer \n";
        std::cout << " * the max length of noise \n";
        std::cout << " * number of training itterations \n";

        throw std::invalid_argument("Incorrect number of arguments.");
    }

    int neuron_count = std::atoi(argv[1]);
    int L_max = std::atoi(argv[2]);
    int epoch_max = std::atoi(argv[3]);

    std::cout << " * number of neurons in recurrent layer : " <<  neuron_count << "\n";
    std::cout << " * the max length of noise              : " << L_max << "\n";
    std::cout << " * number of training itterations       : " << epoch_max <<  "\n";

    return std::make_tuple(neuron_count, L_max, epoch_max);
}

std::vector<dy::real> create_data(vector<dy::real> X, unsigned int L){
    std::random_device r;
    std::default_random_engine eng(r());
    std::uniform_real_distribution rng(0.0, 1.1);

    vector<dy::real> out;

    for(auto x: X){
        out.push_back(x);
        for(auto i = 0; i < L; i++){
            out.push_back(rng(eng));
        }
    }

    return(out);
}

//void train(model, int epoch_max, int[] X, int[] Y){
//}


int main(int argc, char* argv[]){
    //int argc; char** argv;
    dy::realinitialize(argc, argv);

    auto [neuron_count, L_max, epoch_max] = read_args(argc, argv);
    //unsigned int neuron_count, L_max, epoch_max; tie(neuron_count, L_max, epoch_max) = make_tuple(10, 20 , 30);

    vector<dy::real> X = {0.0, 1.0};
    vector<dy::real> Y = {0.0, 1.0};


    for(int epoch = 1; epoch < epoch_max; epoch ++ ){
        ParameterCollection m;
        SimpleSGDTrainer trainer(m, 0.5);
    }
    
}
