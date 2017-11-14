#include <iostream>
#include <chrono>

#define MAT_ROWS 2
#define MAT_COLS 2

int main() {
    int matA[MAT_ROWS][MAT_COLS];
    int matB[MAT_ROWS][MAT_COLS];
    int matC[MAT_ROWS][MAT_COLS];

    srand(time(NULL));

    //Fill arrays with random values between 1 and 20
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            int valA = rand() % 20 + 1;
            int valB = rand() % 20 + 1;
            matA[i][j] = valA;
            matB[i][j] = valB;
        }
    }

    //Do the multiplication C = A * B (naive implementation)
    auto sTime = std::chrono::high_resolution_clock::now(); //start timer
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            for(int k = 0; k < MAT_COLS; k++){
                std::cout << matA[i][k] << " | " << matB[k][j] << std::endl; //doesn't work without this statement for some reason
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
    auto eTime = std::chrono::high_resolution_clock::now(); //stop timer
    std::chrono::duration<double> naiveTime = eTime - sTime;

    //Display the results
    std::cout << "Matrix A" << std::endl;
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            std::cout << matA[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << "Matrix B" << std::endl;
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            std::cout << matB[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << "Result Matrix" << std::endl;
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            std::cout << matC[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << "Computation time: " << naiveTime.count() << "s" << std::endl;

    return 0;
}