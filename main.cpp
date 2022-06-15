#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <thread>

using namespace std;

const int INF = 1e9;
const int MAX_DIM = 5000 + 10;
const int MAX_THREADS = 10;

int sample_size = 100;
int dim = 5000;
float sparsity = 0.8;
int sparsity_steps = 10;
int num_thread = 4;
thread threads[10];

int nnz = INF;
int vec[MAX_DIM];
int mat[MAX_DIM][MAX_DIM];
int ans[MAX_DIM];
int coo_rows[MAX_DIM * MAX_DIM], coo_cols[MAX_DIM * MAX_DIM], coo_vals[MAX_DIM * MAX_DIM];
int csr_offsets[MAX_DIM], csr_cols[MAX_DIM * MAX_DIM], csr_vals[MAX_DIM * MAX_DIM];
int ell_cols[MAX_DIM * MAX_DIM], ell_vals[MAX_DIM * MAX_DIM], max_el = -INF;
int dia_diags[MAX_DIM], dia_vals[2 * MAX_DIM * MAX_DIM], n_diag = 0;


float frand(){
    return (float) rand() / RAND_MAX;
}
int dgrand(){
    return rand() % 9 + 1;
}

void fill_arrays(){
    for (int i = 0; i < dim; ++i){
        if (frand() < sparsity)
            vec[i] = 0;
        else
            vec[i] = dgrand();
    }

    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j){
            if (frand() < sparsity)
                mat[i][j] = 0;
            else
                mat[i][j] = dgrand();
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

void generate_COO_format(bool vec_opt){
    nnz = 0;
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            if (mat[i][j])
                if (!vec_opt || vec[j]){
                    coo_rows[nnz] = i;
                    coo_cols[nnz] = j;
                    coo_vals[nnz] = mat[i][j];
                    ++nnz;
                }
}
void COO_multiplication(){
    for (int i = 0; i < nnz; ++i)
        ans[coo_rows[i]] += coo_vals[i] * vec[coo_cols[i]];
}

void generate_CSR_format(bool vec_opt){
    nnz = 0;

    csr_offsets[0] = 0;
    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j)
            if (mat[i][j])
                if (!vec_opt || vec[j]){
                    csr_cols[nnz] = j;
                    csr_vals[nnz] = mat[i][j];
                    ++nnz;
                }

        csr_offsets[i + 1] = nnz;
    }
}
void CSR_multiplication(int start_row = 0, int end_row = dim){
    for (int r = start_row; r < end_row; ++r)
        for (int i = csr_offsets[r]; i < csr_offsets[r + 1]; ++i)
            ans[r] += csr_vals[i] * vec[csr_cols[i]];
}


void generate_ELL_format(bool vec_opt){
    max_el = -INF;
    for (int i = 0; i < dim; ++i){
        int cnt = 0;
        for (int j = 0; j < dim; ++j)
            if (mat[i][j])
                if (!vec_opt || vec[j])
                    ++cnt;

        max_el = max(max_el, cnt);
    }

    for (int i = 0; i < max_el * dim; ++i){
        ell_cols[i] = -1;
        ell_vals[i] = -1;
    }

    for (int i = 0; i < dim; ++i){
        int cnt = 0;
        for (int j = 0; j < dim; ++j)
            if (mat[i][j])
                if (!vec_opt || vec[j]){
                    ell_cols[i * max_el + cnt] = j;
                    ell_vals[i * max_el + cnt] = mat[i][j];
                    ++cnt;
                }
    }
}
void ELL_multiplication(){
    for (int r = 0; r < dim; ++r)
        for (int i = 0; i < max_el; ++i){
            int ind = r * max_el + i;
            int col = ell_cols[ind];
            if (col != -1)
                ans[r] += ell_vals[ind] * vec[col];
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
            dia_vals[i * n_diag + j] = -1;
        for (int i = 0; i < d; ++i)
            dia_vals[(dim - i - 1) * n_diag + j] = -1;

        for (int i = max(0, -d); i < min(dim, dim - d); ++i)
            dia_vals[i * n_diag + j] = mat[i][i + d];
    }
}
void DIA_multiplication(){
    for (int i = 0; i < dim; ++i)
        for (int d = 0; d < n_diag; ++d){
            int col = i + dia_diags[d];
            int ind = i * n_diag + d;

            if (dia_vals[ind] != -1)
                ans[i] += dia_vals[ind] * vec[col];
        }
}

void general_format_generator(string spmv_format, bool vec_opt){
    if (spmv_format == "COO")
        generate_COO_format(vec_opt);
    else if (spmv_format == "CSR" || spmv_format == "MCSR")
        generate_CSR_format(vec_opt);
    else if (spmv_format == "ELL")
        generate_ELL_format(vec_opt);
    else if (spmv_format == "DIA")
        generate_DIA_format(vec_opt);
}
void general_multiplication(string spmv_format){
    if (spmv_format == "COO")
        COO_multiplication();
    else if (spmv_format == "CSR")
        CSR_multiplication();
    else if (spmv_format == "MCSR"){
        for (int i = 0; i < num_thread; ++i)
            threads[i] = thread(CSR_multiplication, i * dim / num_thread, (i + 1) * dim / num_thread);
        for (int i = 0; i < num_thread; ++i)
            threads[i].join();
    }
    else if (spmv_format == "ELL")
        ELL_multiplication();
    else if (spmv_format == "DIA")
        DIA_multiplication();
    else
        full_multiplication();
}


int main() {
    cout << "Please choose the SpMV format that you want: ";
    string spmv_format;
    cin >> spmv_format;

    cout << "Do you want the vector elements to be used in pre-processing?[y/n] ";
    string yn;
    cin >> yn;
    bool vec_opt = (yn == "y");

    srand(time(0));

    for (int s = 0; s < sparsity_steps; ++s){
        sparsity = double(s) / sparsity_steps;

        int sum_time = 0;
        for (int i = 0; i < sample_size; ++i) {
            fill_arrays();
            general_format_generator(spmv_format, vec_opt);

            clock_t begin_time = clock();
            general_multiplication(spmv_format);
            clock_t end_time = clock();

            sum_time += end_time - begin_time;
        }

        // cout << sparsity << " " << double(sum_time) / sample_size << " clocks" << endl;
        cout << double(sum_time) / sample_size << endl;
    }

    return 0;
}
