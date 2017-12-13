#/bin/bash
rm main & g++ main.cpp -o main -std=c++11 -fopenmp -mavx2 && ./main   
