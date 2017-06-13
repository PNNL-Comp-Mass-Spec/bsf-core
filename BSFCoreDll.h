// BSFCoreDLL.h
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef BSF_CORE_DLL_H__
#define BSF_CORE_DLL_H__

#define NUM_THREADS 8

namespace BSF
{
    class BSFCore
    {
    public:
        static unsigned query(const uint64_t** lib, const uint64_t** q, unsigned** c, const unsigned nlib, const unsigned nq, const unsigned nrow);
        template<typename T>
        static unsigned queryAND(const uint64_t** lib, const uint64_t** q, T** c, const unsigned nlib, const unsigned nq, const unsigned nrow)
	    {
	        unsigned size = (nrow / 64);
	        unsigned k, i, j;
	        uint64_t _xor;
          	unsigned count = 0;

	        #ifdef _OPENMP
	        printf("==================OPENMP====================\n");
	        #pragma omp parallel num_threads(NUM_THREADS)
	        #pragma omp for private(_xor, i, j) schedule(dynamic) reduction(+:count)
	        #endif
	        for( k = 0; k < nlib; k++) {
	            for ( i = 0; i < nq; i++) {
	            	count = 0;
	                for ( j = 0; j < size; j++) {
	                    _xor = (q[i][j]) & (lib[k][j]);
	                    count += __builtin_popcountll(_xor);
	                }
                  c[k][i] = count;
	            }
	        }
	        return 0;
	    }
        // static unsigned queryAND(const uint64_t** lib, const uint64_t** q, T** c, const unsigned nlib, const unsigned nq, const unsigned nrow);
        static unsigned queryXOR(const uint64_t** lib, const uint64_t** q, unsigned** c, const unsigned nlib, const unsigned nq, const unsigned nrow);
        static unsigned analysis(const uint64_t** lib, unsigned** c, const unsigned nlib, const unsigned nrow);
        static unsigned analysis_with_chunks(const uint64_t** lib, unsigned** c, const unsigned x1, const unsigned x2, const unsigned nrow);
        static unsigned analysis_with_chunks(const uint64_t** lib, unsigned** c, const unsigned x1, const unsigned x2, const unsigned y1, const unsigned y2, const unsigned nrow);
        static unsigned analysis_with_query(const uint64_t** lib, const uint64_t** q, unsigned** c, const unsigned x1, const unsigned x2, const unsigned y1, const unsigned y2, const unsigned nrow);
        static unsigned benchmark(const uint64_t** lib, const unsigned nlib, const unsigned nrow);
    };
}
#endif