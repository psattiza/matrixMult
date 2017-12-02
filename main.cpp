#include <iostream>
#include <chrono>
#include <omp.h>
#include <stdio.h>

#define MAT_ROWS 1000
#define MAT_COLS 1000

std::chrono::duration<double> naiveMult(int *mA, int *mB, int *result){
    //Do the multiplication C = A * B (naive implementation)
    auto sTime = std::chrono::steady_clock::now(); //start timer
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            for(int k = 0; k < MAT_ROWS; k++){
                result[i*MAT_COLS+j] += mA[i*MAT_COLS+k] * mB[k*MAT_COLS+j];
            }
        }
    }
    auto eTime = std::chrono::steady_clock::now(); //stop timer
    std::chrono::duration<double> naiveTime = eTime - sTime;
    return naiveTime;
}

std::chrono::duration<double> naiveMultTranspose(int *mA, int *mB, int *result){
    //Do the multiplication C = A * B transposed (naive implementation)
    auto sTime = std::chrono::steady_clock::now(); //start timer
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            for(int k = 0; k < MAT_ROWS; k++){
                //printf("A[%d][%d]=%d  B[%d][%d]=%d\n", i,k,mA[i*MAT_COLS+k] , j, k,mB[MAT_COLS*j+k]);
                result[i*MAT_COLS+j] += mA[i*MAT_COLS+k] * mB[MAT_COLS*j+k];
            }
        }
    }
    auto eTime = std::chrono::steady_clock::now(); //stop timer
    std::chrono::duration<double> naiveTime = eTime - sTime;
    return naiveTime;
}

std::chrono::duration<double> parallelNaiveMult(int *mA, int *mB, int *result){
    //Do the multiplication C = A * B (parallel naive implementation)
    auto sTime = std::chrono::steady_clock::now(); //start timer
    #pragma omp parrallel for
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            for(int k = 0; k < MAT_ROWS; k++){
                result[i*MAT_COLS+j] += mA[i*MAT_COLS+k] * mB[k*MAT_COLS+j];
            }
        }
    }
    auto eTime = std::chrono::steady_clock::now(); //stop timer
    std::chrono::duration<double> naiveTime = eTime - sTime;
    return naiveTime;
}

std::chrono::duration<double> tiledMult(int *mA, int *mB, int *result){
    //Do the multiplication C = A * B (tiled implementation)
    int tileSize = 64;
    auto sTime = std::chrono::steady_clock::now(); //start timer
    #pragma omp parallel for
    for(int i = 0; i < MAT_ROWS; i+=tileSize){
        for(int j = 0; j < MAT_COLS; j+=tileSize){
            for(int k = 0; k < MAT_ROWS; k+=tileSize){
                for(int x = i; x < std::min(i + tileSize, MAT_ROWS); x++){
                    for(int y = j; y < std::min(j + tileSize, MAT_COLS); y++){
                        for(int z = k; z < std::min(k + tileSize, MAT_ROWS); z++){
                            result[x*MAT_COLS+y] += mA[x*MAT_COLS+z] * mB[z*MAT_COLS+y];
                        }
                    }
                }
            }
        }
    }
    auto eTime = std::chrono::steady_clock::now(); //stop timer
    std::chrono::duration<double> naiveTime = eTime - sTime;
    return naiveTime;
}

std::chrono::duration<double> tiledMultTranspose(int *mA, int *mB, int *result){
    //Do the multiplication C = A * B (tiled implementation)
    int tileSize = 64;
    auto sTime = std::chrono::steady_clock::now(); //start timer
    #pragma omp parallel for schedule(guided)
    for(int i = 0; i < MAT_ROWS; i+=tileSize){
        for(int j = 0; j < MAT_COLS; j+=tileSize){
            for(int k = 0; k < MAT_ROWS; k+=tileSize){
                for(int x = i; x < std::min(i + tileSize, MAT_ROWS); x++){
                    for(int y = j; y < std::min(j + tileSize, MAT_COLS); y++){
                        for(int z = k; z < std::min(k + tileSize, MAT_ROWS); z++){
                            result[x*MAT_COLS+y] += mA[x*MAT_COLS+z] * mB[z+MAT_COLS*y];
                        }
                    }
                }
            }
        }
    }
    auto eTime = std::chrono::steady_clock::now(); //stop timer
    std::chrono::duration<double> naiveTime = eTime - sTime;
    return naiveTime;
}

std::chrono::duration<double> customMult(int *mA, int *mB, int *result){
    //Do the multiplication C = A * B (custom implementation)
    int tileSize = 64;
    auto sTime = std::chrono::steady_clock::now(); //start timer
    for(int i = 0; i < MAT_ROWS; i+=tileSize){
        for(int j = 0; j < MAT_COLS; j+=tileSize){
            for(int k = 0; k < MAT_ROWS; k+=tileSize){
                for(int x = i; x < std::min(i + tileSize, MAT_ROWS); x++){
                    for(int y = j; y < std::min(j + tileSize, MAT_COLS); y++){
                        for(int z = k; z < std::min(k + tileSize, MAT_ROWS); z++){
                            result[x*MAT_COLS+y] += mA[x*MAT_COLS+z] * mB[z*MAT_COLS+y];
                        }
                    }
                }
            }
        }
    }
    auto eTime = std::chrono::steady_clock::now(); //stop timer
    std::chrono::duration<double> naiveTime = eTime - sTime;
    return naiveTime;
}

void displayMatrices(int *mA, int *mB, int *result){
    std::cout << "Matrix A" << std::endl;
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            std::cout << mA[i*MAT_COLS+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << "Matrix B" << std::endl;
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            std::cout << mB[i*MAT_COLS+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << "Result Matrix" << std::endl;
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            std::cout << result[i*MAT_COLS+j] << " ";
        }
        std::cout << std::endl;
    }
}

bool compare2(int *a, int *b){
    for(int row = 0; row < MAT_ROWS; row++)
        for(int col = 0; col < MAT_COLS; col++)
            if(a[row*MAT_ROWS + col] != b[row*MAT_ROWS + col])        
                return false;
    return true;
}

bool printMatrix(int *a){
    for(int row = 0; row < MAT_ROWS; row++){
        for(int col = 0; col < MAT_COLS; col++)
            printf("%4d ", a[row*MAT_ROWS + col]);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    omp_set_num_threads(4);
    int *matA = new int[MAT_ROWS*MAT_COLS];
    int *matB = new int[MAT_ROWS*MAT_COLS];
    int *matC = new int[MAT_ROWS*MAT_COLS];
    int *matC1 = new int[MAT_ROWS*MAT_COLS];
    int *matC2 = new int[MAT_ROWS*MAT_COLS];
    int *matC3 = new int[MAT_ROWS*MAT_COLS];
    int *matC4 = new int[MAT_ROWS*MAT_COLS];
    int *matC5 = new int[MAT_ROWS*MAT_COLS];
    int *matBTrans = new int[MAT_ROWS*MAT_COLS];

    srand(time(NULL));

    std::cout << "Initializing matrices..." << std::endl;

    //Fill matrices with random values between 1 and 20, or 0 for result matrix
    #pragma omp parallel for schedule(guided)
    for(int i = 0; i < MAT_ROWS; i++){
        for(int j = 0; j < MAT_COLS; j++){
            int valA = rand() % 20 + 1;
            int valB = rand() % 20 + 1;
            matA[i*MAT_COLS+j] = valA;
            matB[i*MAT_COLS+j] = valB;
            matBTrans[j*MAT_COLS+i] = valB;
            matC[i*MAT_COLS+j] = 0;
        }
    }

    std::cout << "Performing multiplications..." << std::endl;

    //Run and time the naive implementation
    std::chrono::duration<double> naiveTime = naiveMult(matA, matB, matC);
    std::chrono::duration<double> naiveTransposeTime = naiveMultTranspose(matA, matBTrans, matC4);

    //Run and time the parallel naive implementation
    std::chrono::duration<double> parallelNaiveTime = parallelNaiveMult(matA, matB, matC1);

    //Run and time tiled implementation
    std::chrono::duration<double> tiledTime = tiledMult(matA, matB, matC2);
    std::chrono::duration<double> tiledTimeTranspose = tiledMultTranspose(matA, matBTrans, matC5);

    //Run and time our implementation
    std::chrono::duration<double> customTime = customMult(matA, matB, matC3);


    std::cout << std::endl << Validating..." << std::endl;
    std::cout << std::boolalpha << "Transpose validating: " << compare2(matC, matC4) << std::endl;
    //printMatrix(matA);
    //printMatrix(matB);
    //printMatrix(matBTrans);
    //printMatrix(matC);
    //printMatrix(matC4);
    std::cout << std::boolalpha << "Parallel validating: " << compare2(matC, matC1) << std::endl;
    std::cout << std::boolalpha << "Tiled validating: " << compare2(matC, matC2) << std::endl;
    std::cout << std::boolalpha << "Tiled transpose validating: " << compare2(matC, matC5) << std::endl;
    std::cout << std::boolalpha << "Custom validating: " << compare2(matC, matC3) << std::endl;

    //Display the results
    //displayMatrices(matA, matB, matC);
    std::cout << std::endl << "Computation time (naive): " << naiveTime.count() << "s" << std::endl;
    std::cout << "Computation time (transpose naive): " << naiveTransposeTime.count() << "s" << std::endl;
    std::cout << "Computation time (parallel naive): " << parallelNaiveTime.count() << "s" << std::endl;
    std::cout << "Computation time (tiled): " << tiledTime.count() << "s" << std::endl;
    std::cout << "Computation time (transposed tiled): " << tiledTimeTranspose.count() << "s" << std::endl;
    std::cout << "Computation time (our implementation): " << customTime.count() << "s" << std::endl;

   return 0;
}
