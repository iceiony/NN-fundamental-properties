#pragma once 

#include "dynet/expr.h"
#include <vector>

namespace dy = dynet;
using namespace dynet;
using namespace std;

class RNN {
    private:
        Parameter hidden;
        Parameter output;
        unsigned state_size;
    public: 
        ParameterCollection params;

        RNN(unsigned neuron_count){
            state_size = (unsigned) neuron_count;
            hidden = params.add_parameters({state_size, 1 + state_size});
            output = params.add_parameters({1, state_size});
        }

        Expression forward(ComputationGraph& cg, Expression x){
            auto h = dy::zeroes(cg, Dim({state_size}, x.dim().batch_elems()));

            Expression w = parameter(cg, hidden);

            auto size = x.dim().rows();
            for(unsigned i = 0; i < size; i++){
                 h = concatenate({ dy::pick(x, i), h });
                 h = tanh(w * h);
            }

            w = parameter(cg, output);
            return logistic(w * h);
        }
};
