
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>

#define LM 6
#define LN 6
#define LP 6

#define M ( 1 << LM )
#define N ( 1 << LN )
#define P ( 1 << LP )


 void multiply_Arrays_hw(uint8_t a[M][N], uint8_t b[N][P], uint32_t result[M][P]){
    
	uint8_t bram_a[N][M];
	uint8_t bram_b[M][P];

	// Config for the BRAMS // 
	#pragma HLS ARRAY_PARTITION variable=bram_a cyclic factor=8 dim=1
	#pragma HLS ARRAY_PARTITION variable=bram_b cyclic factor=8 dim=2
	#pragma HLS array_partition variable=result cyclic factor=32 dim=2

	// Load the a array to the bram_a
	for (int i = 0 ; i < N ; i++){
		for(int j = 0 ; j < M ; j++){
			bram_a[i][j] = a[i][j];
		}
	}
	// Load the b array to the bram_a
	for (int i = 0 ; i < M ; i++){
		for(int j = 0 ; j < P ; j++){
			bram_b[i][j] = b[i][j];
		}
	}
	// Multiplication Algorithm
	#pragma HLS PIPELINE
	for(int i = 0; i < N; i++){
		for(int j = 0; j < P; j++){
            uint32_t r = 0;
			#pragma HLS unroll factor=32
            for(int k = 0; k < M; k++){
            	r += bram_a[i][k] * bram_b[k][j];
            }
            result[i][j] = r;
        }
    }
 }

