#/bin/bash
rm main & g++ main.cpp -o main -std=c++11 -O1 -fopenmp -mavx2 && ./main   
