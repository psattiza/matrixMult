#include <iostream>
#include <chrono>
#include <omp.h>

#define MAT_ROWS 250
#define MAT_COLS 250

std::chrono::duration<double> naiveMult(int (&mA)[MAT_ROWS][MAT_COLS], int (&mB)[MAT_ROWS][MAT_COLS], int (&result)[MAT_ROWS][MAT_COLS]){
    //Do the multiplication C = A * B (naive implementation)
    auto sTime = std::chrono::steady_clock::now(); //start timer
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            for(int k = 0; k < MAT_COLS; k++){
                result[i][j] += mA[i][k] * mB[k][j];
            }
        }
    }
    auto eTime = std::chrono::steady_clock::now(); //stop timer
    std::chrono::duration<double> naiveTime = eTime - sTime;
    return naiveTime;
}

std::chrono::duration<double> parallelNaiveMult(int (&mA)[MAT_ROWS][MAT_COLS], int (&mB)[MAT_ROWS][MAT_COLS], int (&result)[MAT_ROWS][MAT_COLS]){
    //Do the multiplication C = A * B (parallel naive implementation)
    auto sTime = std::chrono::steady_clock::now(); //start timer
    #pragma omp parrallel for
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            for(int k = 0; k < MAT_COLS; k++){
                result[i][j] += mA[i][k] * mB[k][j];
            }
        }
    }
    auto eTime = std::chrono::steady_clock::now(); //stop timer
    std::chrono::duration<double> naiveTime = eTime - sTime;
    return naiveTime;
}

void displayMatrices(int (&mA)[MAT_ROWS][MAT_COLS], int (&mB)[MAT_ROWS][MAT_COLS], int (&result)[MAT_ROWS][MAT_COLS]){
    std::cout << "Matrix A" << std::endl;
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            std::cout << mA[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << "Matrix B" << std::endl;
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            std::cout << mB[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << "Result Matrix" << std::endl;
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            std::cout << result[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int matA[MAT_ROWS][MAT_COLS];
    int matB[MAT_ROWS][MAT_COLS];
    int matC[MAT_ROWS][MAT_COLS];

    srand(time(NULL));

    //Fill matrices with random values between 1 and 20, or 0 for result matrix
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            int valA = rand() % 20 + 1;
            int valB = rand() % 20 + 1;
            matA[i][j] = valA;
            matB[i][j] = valB;
            matC[i][j] = 0;
        }
    }

    //Run and time the naive implementation
    std::chrono::duration<double> naiveTime = naiveMult(matA, matB, matC);

    //Run and time the parallel naive implementation
    std::chrono::duration<double> parallelNaiveTime = parallelNaiveMult(matA, matB, matC);

    //Display the results
    //displayMatrices(matA, matB, matC);
    std::cout << std::endl << "Computation time (naive): " << naiveTime.count() << "s" << std::endl;
    std::cout << std::endl << "Computation time (parallel naive): " << parallelNaiveTime.count() << "s" << std::endl;

   return 0;
}