#include <tuple>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace std;

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

int main(int argc, char* argv[]){

    std::random_device r;
    std::default_random_engine eng(r());
    std::uniform_real_distribution rng(0.0, 1.1);

    auto [neuron_count, L_max, epoch_max] = read_args(argc, argv);

    double X[] = {0.0, 1.1};
}
