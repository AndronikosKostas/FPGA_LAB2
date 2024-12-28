//OUR CODE

#define lm 4
#define ln 4
#define lp 4

#define m  (1 << lm)
#define n  (1 << ln)
#define p  (1 << lp)

// opencl pipa
extern "C"
{
	void MATRIX_MUL_HW(int A[n][m], int B[m][p], int C[n][p])
	{

	// για την επικοινωνια του kernel και του host 
	#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem
	#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem
	#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem
	#pragma HLS INTERFACE s_axilite port = A bundle = control
	#pragma HLS INTERFACE s_axilite port = B bundle = control
	#pragma HLS INTERFACE s_axilite port = C bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control


	int BRAM_A[n][m];
	int BRAM_B[m][p];

	#pragma HLS ARRAY_PARTITION variable=BRAM_A cyclic factor=32 dim=2
	#pragma HLS ARRAY_PARTITION variable=BRAM_B cyclic factor=32 dim=1

//Transfer data from DRAM to the partitioned memory
		for (int i = 0 ; i < n ; i++){
			// NO pipeline HERE
			for(int j = 0 ; j < m ; j++){
				BRAM_A[i][j] = A[i][j];
			}
		}

		for (int i = 0 ; i < m ; i++){
			// NO pipeline HERE
			for(int j = 0 ; j < p ; j++){
				BRAM_B[i][j] = B[i][j];
			}
		}

//Perform matrix multiplication
		for(int i = 0; i < n; i++){
			for(int j = 0; j < p; j++){
				int res = 0;
				//#pragma HLS PIPELINE
				#pragma HLS loop_tripcount min=64 max=64 avg=64
					for(int k = 0; k < m; k++){
						#pragma HLS unroll factor=64
						res += BRAM_A[i][k] * BRAM_B[k][j];
					}
				C[i][j] = res;
			}
		}
	}
}

//END
