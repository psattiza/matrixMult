#include <iostream>
#include <chrono>
#include <omp.h>
#include <immintrin.h>

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

std::chrono::duration<double> customMult(int (&mA)[MAT_ROWS][MAT_COLS], int (&mB)[MAT_ROWS][MAT_COLS], int (&result)[MAT_ROWS][MAT_COLS]){
    //Do the multiplication C = A * B (custom implementation using tiling and special AVX instructions)
    //int tileSize = 64;
    auto sTime = std::chrono::steady_clock::now(); //start timer
    /*#pragma omp parallel for schedule(guided)
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
    }*/
    unsigned int mc = MAT_COLS;
    size_t s1 = std::min(512u, mc);
    size_t s2 = std::min(24u, mc);

    for(size_t jj = 0; jj < mc; jj+=s1){
        for(size_t kk = 0; kk < mc; kk+=s2){
            //loop unrolling by factor of two
            for(size_t i = 0; i < mc; i+=2){
                for(size_t j = jj; j < jj + s1; j += 16){
                    __m256i sumA1, sumA2, sumB1, sumB2;
                    if(kk == 0){
                        sumA1 = sumA2 = sumB1 = sumB2 = _mm256_setzero_si256();
                    }
                    else{
                        sumA1 = _mm256_load_si256((__m256i*)&result[i][j]);
                        sumB2 = _mm256_load_si256((__m256i*)&result[i][j+8]);
                        sumA2 = _mm256_load_si256((__m256i*)&result[i+1][j]);
                        sumB2 = _mm256_load_si256((__m256i*)&result[i+1][j+8]);
                    }
                    size_t limit = std::min((size_t )mc, kk + s2);
                    for(size_t k = kk; k < limit; k++){
                        auto bc_mat11 = _mm256_set1_epi32(mA[i][k]);
                        auto vecA_mat2 = _mm256_loadu_si256((__m256i*)&mB[k][j]);
                        auto vecB_mat2 = _mm256_loadu_si256((__m256i*)&mB[k][j+8]);
                        sumA1 = _mm256_add_epi32(sumA1, _mm256_mullo_epi32(bc_mat11, vecA_mat2));
                        sumB1 = _mm256_add_epi32(sumB1, _mm256_mullo_epi32(bc_mat11, vecB_mat2));
                        auto bc_mat12 = _mm256_set1_epi32(mA[i+1][k]);
                        sumA2 = _mm256_add_epi32(sumA2, _mm256_mullo_epi32(bc_mat12, vecA_mat2));
                        sumB2 = _mm256_add_epi32(sumB2, _mm256_mullo_epi32(bc_mat12, vecB_mat2));
                    }
                    _mm256_storeu_si256((__m256i*)&result[i][j], sumA1);
                    _mm256_storeu_si256((__m256i*)&result[i][j+8], sumB1);
                    _mm256_storeu_si256((__m256i*)&result[i][j+8], sumA2);
                    _mm256_storeu_si256((__m256i*)&result[i+1][j+8], sumB2);
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

int main() {
    std::cout << "Declaring old arrays..." << std::endl;
    int *matA = new int[MAT_ROWS*MAT_COLS];
    int *matB = new int[MAT_ROWS*MAT_COLS];
    int *matC = new int[MAT_ROWS*MAT_COLS];
    int *matC1 = new int[MAT_ROWS*MAT_COLS];
    int *matC2 = new int[MAT_ROWS*MAT_COLS];
    int *matC3 = new int[MAT_ROWS*MAT_COLS];
    int *matBTrans = new int[MAT_ROWS*MAT_COLS];

    std::cout << "Declaring new arrays..." << std::endl;

    int matA2d[MAT_ROWS][MAT_COLS];
    int matB2d[MAT_ROWS][MAT_COLS];
    int matC2d[MAT_ROWS][MAT_COLS] = {{0}};

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
            matA2d[i][j] = valA;
            matB2d[i][j] = valB;
            //matC2d[i][j] = 0;
        }
    }

    std::cout << "Performing multiplications..." << std::endl;

    //Run and time the naive implementation
    std::cout << "Using naive..." << std::endl;
    std::chrono::duration<double> naiveTime = naiveMult(matA, matB, matC);
    std::cout << "Computation time (naive): " << naiveTime.count() << "s" << std::endl;

    //Run and time the parallel naive implementation
    std::cout << "Using parallel naive..." << std::endl;
    std::chrono::duration<double> parallelNaiveTime = parallelNaiveMult(matA, matB, matC1);
    std::cout << "Computation time (parallel naive): " << parallelNaiveTime.count() << "s" << std::endl;

    //Run and time tiled implementation
    std::cout << "Using tiled..." << std::endl;
    std::chrono::duration<double> tiledTime = tiledMult(matA, matB, matC2);
    std::cout << "Computation time (tiled): " << tiledTime.count() << "s" << std::endl;

    //Run and time our implementation
    //std::cout << "Using custom..." << std::endl;
    //std::chrono::duration<double> customTime = customMult(matA2d, matB2d, matC2d);
    //std::cout << "Computation time (our implementation): " << customTime.count() << "s" << std::endl;

    std::cout << std::boolalpha << "N v P " << compare2(matC, matC1) << std::endl;
    std::cout << std::boolalpha << "N v T " << compare2(matC, matC2) << std::endl;
    //std::cout << std::boolalpha << "N v C " << compare2(matC, matC3) << std::endl;

    //Display the results
    //displayMatrices(matA, matB, matC);

   return 0;
}
