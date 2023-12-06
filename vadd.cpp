#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <cstdint>

#define LM 4
#define LN 4
#define LP 4

#define M ( 1 << LM )
#define N ( 1 << LN )
#define P ( 1 << LP )



extern "C" {
	void vadd(int *a, int *b, int *result){


		#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem
		#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmem
		#pragma HLS INTERFACE m_axi port = result offset = slave bundle = gmem
		#pragma HLS INTERFACE s_axilite port = a bundle = control
		#pragma HLS INTERFACE s_axilite port = b bundle = control
		#pragma HLS INTERFACE s_axilite port = result bundle = control
		#pragma HLS INTERFACE s_axilite port = return bundle = control

		int bram_a[M][N];
		int bram_b[N][P];


		#pragma HLS ARRAY_PARTITION variable=bram_a cyclic factor=8 dim=2
		#pragma HLS ARRAY_PARTITION variable=bram_b cyclic factor=8 dim=1



			for (int i = 0 ; i < M ; i++){

				for(int j = 0 ; j < N ; j++){
					#pragma HLS PIPELINE IT=1
					bram_a[i][j] = a[(i*N) + j];
				}
			}

			for (int i = 0 ; i < N ; i++){
				for(int j = 0 ; j < P ; j++){
					#pragma HLS PIPELINE IT=1
					bram_b[i][j] = b[(i*N) + j];
				}
			}


			for(int i = 0; i < M; i++){
				for(int j = 0; j < P; j++){
		            int r = 0;
					#pragma HLS PIPELINE IT=1
		            for(int k = 0; k < N; k++){
					   	r += bram_a[i][k] * bram_b[k][j];
		            }
		            result[(i*M) + j] = r;
		        }
		    }
    }

}