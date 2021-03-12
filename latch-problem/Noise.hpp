#pragma one

#include <random>
#include <vector>
#include "dynet/expr.h"

using namespace std;
namespace dy = dynet;

class Noise {
    private:
        std::random_device r;
        std::default_random_engine eng;
        std::uniform_real_distribution<float> rng;
    public:
        Noise(float min, float max){
            eng = std::default_random_engine(r());
            rng = std::uniform_real_distribution(min, max);
        }

        std::vector<dy::real> create_data(vector<dy::real> X, unsigned int L){
            vector<dy::real> out;

            for(auto x: X){
                out.push_back(x);
                for(auto i = 0; i < L; i++){
                    out.push_back(rng(eng));
                }
            }

            return(out);
        }
};
