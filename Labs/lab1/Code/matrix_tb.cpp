#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>

#include <cstdint>

#define LM 1
#define LN 1
#define LP 1

#define M  (1 << LM)   
#define N  (1 << LN)
#define P  (1 << LP)

using namespace std;

void multiply_Arrays_sw(uint8_t a[M][N], uint8_t b[N][P], uint32_t result[M][P]);
void initialize_Arrays(uint8_t a[M][N], uint8_t b[N][P]);

template <typename T>
void print_Array(T a[M][N]){
    for(int i = 0 ; i < M ; i++){
        for(int j = 0 ; j < N; j++){
           printf("%d", a[i][j]);
            printf("\t");
        }
        printf("\n");
    } 
}

bool checkEquality(uint32_t r_sw[M][P], uint32_t r_hw[M][P]);




int main(){
    // create the test arrays 
    uint8_t a[M][N], b[N][P];
    uint32_t result_sw[M][P], result_hw[M][P];
    // init the arrays with random values 
    initialize_Arrays(a,b);
    // print the array a
    print_Array(a);
    printf("\n");
    // print the array b
    print_Array(b);
    printf("\n");
    printf("\n");
    printf("\n");
    // multiplication from software 
    multiply_Arrays_sw(a,b,result_sw);
    // multiplication from hardware 
    // multiply_Arrays_hw(a,b,result_hw);
    printf("Result array from SW \n");
    print_Array(result_sw);
    // printf("Result array from HW \n");
    // print_Array(result_hw);
    // check if the results arrays are the same 
    bool same = checkEquality(result_sw, result_sw);
    if(same == true)
        printf("TEST PASSED\n");
    else
        printf("TEST FAILED\n");

    return 0;
}


void multiply_Arrays_sw(uint8_t a[M][N], uint8_t b[N][P], uint32_t result[M][P]){
    for(int i = 0 ; i < M ; i++){
        for(int j = 0; j < N ; j++){
           result[i][j] = 0;
           for (int k = 0; k < P; ++k)
                result[i][j] += a[i][k] * b[k][j];
        }
    }
}
    

void initialize_Arrays(uint8_t a[M][N], uint8_t b[N][P]){
    for(int i = 0 ; i < M ; i++){
        for(int j = 0 ; j < N ; j++){
           a[i][j] = (uint8_t)(rand() % 256) ;
           b[i][j] = (uint8_t)(rand() % 256) ;
        }
    } 
}


bool checkEquality(uint32_t r_sw[M][P], uint32_t r_hw[M][P]){

    bool b = true;
    for(int i = 0 ; i < M ; i++)
    {
        for(int j = 0; j < P ; j++)
        {
            if(r_sw[i][j] != r_hw[i][j])
                b = false;
        }   
    }
    return b;
}
