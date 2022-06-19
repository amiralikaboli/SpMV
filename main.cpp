#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <thread>
#include <immintrin.h>

using namespace std;

const int INF = 1e9;
const int MAX_DIM = 5000 + 10;
const int MAX_THREADS = 10;

int sample_size = 100;
int dim = 2000;
float sparsity = 0.5;
int sparsity_steps = 10;
int num_thread = 4;
thread threads[10];

int nnz = INF;
float vec[MAX_DIM];
float mat[MAX_DIM][MAX_DIM];
float ans[MAX_DIM];
int coo_rows[MAX_DIM * MAX_DIM], coo_cols[MAX_DIM * MAX_DIM];
float coo_vals[MAX_DIM * MAX_DIM], coo_vecs[MAX_DIM * MAX_DIM];
int csr_offsets[MAX_DIM], csr_cols[MAX_DIM * MAX_DIM];
float csr_vals[MAX_DIM * MAX_DIM], csr_vecs[MAX_DIM * MAX_DIM];
int ell_cols[MAX_DIM * MAX_DIM], max_el = -INF;
float ell_vals[MAX_DIM * MAX_DIM], ell_vecs[MAX_DIM * MAX_DIM];
int dia_diags[MAX_DIM], n_diag = 0;
float dia_vals[2 * MAX_DIM * MAX_DIM];


float rand01(){
    return (float) rand() / RAND_MAX;
}
float float_rand(){
    return (rand() % 100 + 1) / 10.0;
}

void fill_arrays(){
    for (int i = 0; i < dim; ++i){
        if (rand01() < sparsity)
            vec[i] = 0;
        else
            vec[i] = float_rand();
    }

    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j){
            if (rand01() < sparsity)
                mat[i][j] = 0;
            else
                mat[i][j] = float_rand();
        }
    }

    for (int i = 0; i < dim; ++i)
        ans[i] = 0;
}

void full_multiplication(){
    for (int r = 0; r < dim; ++r)
        for (int i = 0; i < dim; ++i)
            ans[r] += mat[r][i] * vec[i];
}

void generate_COO_format(bool vec_opt, bool use_simd){
    nnz = 0;
    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j)
            if (mat[i][j])
                if (!vec_opt || vec[j]){
                    coo_rows[nnz] = i;
                    coo_cols[nnz] = j;
                    coo_vals[nnz] = mat[i][j];
                    coo_vecs[nnz] = vec[j];
                    ++nnz;
                }

        if (use_simd)
            while(nnz % 8 != 0){
                coo_rows[nnz] = MAX_DIM - 1;
                coo_cols[nnz] = MAX_DIM - 1;
                coo_vals[nnz] = 0;
                coo_vecs[nnz] = 0;
                ++nnz;
            }
    }
}
void COO_multiplication(bool use_simd){
    if (use_simd){
        __m256* mcoo_vals = (__m256*) coo_vals;
        __m256* mcoo_vecs = (__m256*) coo_vecs;
        __m256 ymm, ymm2;

        int steps = nnz / 8;
        for (int i = 0; i < steps; ++i){
            ymm = _mm256_mul_ps(mcoo_vals[i], mcoo_vecs[i]);
            ymm2 = _mm256_permute2f128_ps(ymm , ymm , 1);
            ymm = _mm256_add_ps(ymm, ymm2);
            ymm = _mm256_hadd_ps(ymm, ymm);
            ymm = _mm256_hadd_ps(ymm, ymm);

            ans[coo_rows[(i << 3)]] += ymm[0];
        }
    }
    else{
        for (int i = 0; i < nnz; ++i)
            ans[coo_rows[i]] += coo_vals[i] * vec[coo_cols[i]];
    }
}

void generate_CSR_format(bool vec_opt, bool use_simd){
    nnz = 0;

    csr_offsets[0] = 0;
    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j)
            if (mat[i][j])
                if (!vec_opt || vec[j]){
                    csr_cols[nnz] = j;
                    csr_vals[nnz] = mat[i][j];
                    csr_vecs[nnz] = vec[j];
                    ++nnz;
                }

        if (use_simd)
            while (nnz % 8 != 0) {
                csr_cols[nnz] = MAX_DIM - 1;
                csr_vals[nnz] = 0;
                csr_vecs[nnz] = 0;
                ++nnz;
            }

        csr_offsets[i + 1] = nnz;
    }
}
void CSR_multiplication(bool use_simd, int l_row = 0, int r_row = dim){
    if (use_simd){
        for (int r = l_row; r < r_row; ++r){
            __m256* mcoo_vals = (__m256*) (csr_vals + csr_offsets[r]);
            __m256* mcoo_vecs = (__m256*) (csr_vecs + csr_offsets[r]);
            __m256 ymm, ymm2;

            int steps = ((csr_offsets[r + 1] - csr_offsets[r]) >> 3);
            for (int i = 0; i < steps; ++i){
                ymm = _mm256_mul_ps(mcoo_vals[i], mcoo_vecs[i]);
                ymm2 = _mm256_permute2f128_ps(ymm , ymm , 1);
                ymm = _mm256_add_ps(ymm, ymm2);
                ymm = _mm256_hadd_ps(ymm, ymm);
                ymm = _mm256_hadd_ps(ymm, ymm);

                ans[r] += ymm[0];
            }
        }
    }
    else{
        // #pragma omp parallel for
        for (int r = l_row; r < r_row; ++r)
            for (int i = csr_offsets[r]; i < csr_offsets[r + 1]; ++i)
                ans[r] += csr_vals[i] * vec[csr_cols[i]];
    }
}

void generate_ELL_format(bool vec_opt, bool use_simd){
    max_el = -INF;
    for (int i = 0; i < dim; ++i){
        int cnt = 0;
        for (int j = 0; j < dim; ++j)
            if (mat[i][j])
                if (!vec_opt || vec[j])
                    ++cnt;

        max_el = max(max_el, cnt);
    }
    if (use_simd)
        max_el = ((max_el / 8) + 1) * 8;

    for (int i = 0; i < max_el * dim; ++i){
        ell_cols[i] = MAX_DIM - 1;
        ell_vals[i] = 0;
    }

    for (int i = 0; i < dim; ++i){
        int cnt = 0;
        for (int j = 0; j < dim; ++j)
            if (mat[i][j])
                if (!vec_opt || vec[j]){
                    ell_cols[i * max_el + cnt] = j;
                    ell_vals[i * max_el + cnt] = mat[i][j];
                    ell_vecs[i * max_el + cnt] = vec[j];
                    ++cnt;
                }
    }
}
void ELL_multiplication(bool use_simd, int l_row = 0, int r_row = dim){
    if (use_simd){
        for (int r = l_row; r < r_row; ++r){
            __m256* mcoo_vals = (__m256*) (ell_vals + (r * max_el));
            __m256* mcoo_vecs = (__m256*) (ell_vecs + (r * max_el));
            __m256 ymm, ymm2;

            int steps = (max_el >> 3);
            for (int i = 0; i < steps; ++i){
                ymm = _mm256_mul_ps(mcoo_vals[i], mcoo_vecs[i]);
                ymm2 = _mm256_permute2f128_ps(ymm , ymm , 1);
                ymm = _mm256_add_ps(ymm, ymm2);
                ymm = _mm256_hadd_ps(ymm, ymm);
                ymm = _mm256_hadd_ps(ymm, ymm);

                ans[r] += ymm[0];
            }
        }
    }
    else{
        int l_limit = l_row * max_el;
        int r_limit = r_row * max_el;
        for (int i = l_limit; i < r_limit; ++i)
            if (ell_cols[i] != MAX_DIM - 1)
                ans[i / max_el] += ell_vals[i] * vec[ell_cols[i]];
    }
}

void generate_DIA_format(bool vec_opt){
    n_diag = 0;
    for (int d = -(dim - 1); d < dim; ++d){
        bool flag = false;
        for (int i = 0; i < dim; ++i){
            int j = i + d;
            if (j >= 0 && j < dim && mat[i][j])
                if (!vec_opt || vec[j])
                    flag = true;
        }

        if (flag){
            dia_diags[n_diag] = d;
            ++n_diag;
        }
    }

    for (int j = 0; j < n_diag; ++j){
        int d = dia_diags[j];

        for (int i = 0; i < -d; ++i)
            dia_vals[i * n_diag + j] = 0;
        for (int i = 0; i < d; ++i)
            dia_vals[(dim - i - 1) * n_diag + j] = 0;

        for (int i = max(0, -d); i < min(dim, dim - d); ++i)
            dia_vals[i * n_diag + j] = mat[i][i + d];
    }
}
void DIA_multiplication(){
    for (int i = 0; i < dim; ++i)
        for (int d = 0; d < n_diag; ++d){
            int col = i + dia_diags[d];
            int ind = i * n_diag + d;

            if (dia_vals[ind] != 0)
                ans[i] += dia_vals[ind] * vec[col];
        }
}

void general_format_generator(string spmv_format, bool vec_opt, bool use_simd){
    if (spmv_format == "COO")
        generate_COO_format(vec_opt, use_simd);
    else if (spmv_format == "CSR")
        generate_CSR_format(vec_opt, use_simd);
    else if (spmv_format == "ELL")
        generate_ELL_format(vec_opt, use_simd);
    else if (spmv_format == "DIA")
        generate_DIA_format(vec_opt);
}
void general_multiplication(string spmv_format, bool use_simd, int l_row = 0, int r_row = dim){
    if (spmv_format == "COO")
        COO_multiplication(use_simd);
    else if (spmv_format == "CSR")
        CSR_multiplication(use_simd, l_row, r_row);
    else if (spmv_format == "ELL")
        ELL_multiplication(use_simd, l_row, r_row);
    else if (spmv_format == "DIA")
        DIA_multiplication();
    else
        full_multiplication();
}

void general_parallism_controller(bool run_parallel, bool spmv_format bool use_simd){
    if (!run_parallel){
        general_multiplication(spmv_format, use_simd);
        return;
    }

    if (spmv_format == "CSR" || spmv_format == "ELL"){
        for (int i = 0; i < num_thread; ++i)
            threads[i] = thread(
                general_multiplication,
                spmv_format,
                use_simd,
                i * dim / num_thread,
                (i + 1) * dim / num_thread
            );
        for (int i = 0; i < num_thread; ++i)
            threads[i].join();
    }
    else
        general_multiplication(spmv_format, use_simd);
}


int main() {
    cout << "Please choose the SpMV format that you want: ";
    string spmv_format;
    cin >> spmv_format;

    cout << "Do you want the vector elements to be used in pre-processing?[y/n] ";
    string yn;
    cin >> yn;
    bool vec_opt = (yn == "y");

    cout << "Do you want SIMD to be used?[y/n] ";
    cin >> yn;
    bool use_simd = (yn == "y");

    cout << "Do you want to run in parallel?[y/n] ";
    cin >> yn;
    bool run_parallel = (yn == "y");

    srand(time(0));

    for (int s = 0; s < sparsity_steps; ++s){
        sparsity = double(s) / sparsity_steps;

        int sum_time = 0;
        for (int i = 0; i < sample_size; ++i) {
            fill_arrays();
            general_format_generator(spmv_format, vec_opt, use_simd);

            clock_t begin_time = clock();
            general_parallism_controller(run_parallel, spmv_format, use_simd)
            // general_multiplication(spmv_format, use_simd);
            clock_t end_time = clock();

            sum_time += end_time - begin_time;
        }

        // cout << sparsity << " " << double(sum_time) / sample_size << " clocks" << endl;
        cout << double(sum_time) / sample_size << endl;
    }

    return 0;
}
