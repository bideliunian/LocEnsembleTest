/*
   acov.c: angle correlation and covariance statistics
   and screening
   Software: Qi Zhang
*/

#include <R.h>
#include <Rmath.h>
//#include "angle_omp.h"

void   aCOV(double *x, double *y, int *dims, double *ACOV);
double D_center(double **akl, double **A, int n);
double U_center(double **akl, double **A, int n);
void   acorsis(double *x, double *y, int *dims, double *acor_stat); 

/* functions in utilities.c */
double **alloc_matrix(int r, int c);
int    **alloc_int_matrix(int r, int c);
double ***alloc_array(int r, int c, int d);
void   free_matrix(double **matrix, int r, int c);
void   free_array(double ***array, int r, int c, int d);
void   free_int_matrix(int **matrix, int r, int c);
void   vector2matrix(double *x, double **y, int N, int d, int isroworder);
void   distance(double **bxy, double **D, int N, int d);
void   Euclidean_distance(double *x, double **Dx, int n, int d);
void   quicksort(double *array, int *idx, int l, int u);


void aCOV(double *x, double *y, int *dims, double *ACOV) {
    /*  computes aCov(x,y), aCor(x,y), aVar(x), aVar(y)
        V-statistic is n*aCov^2 
        x, y: two n by n distance matrixes
        dims[0] = n (sample size)
        dims[1] = dst (logical, TRUE if x, y are distances)
        ACOV  : vector [aCov, aCor, aVar(x), aVar(y), mean(A), mean(B)]
     */
    int    i, j, k, l, n, n3;
    int    dst;
    double **Dx, **Dy, **Dx2, **Dy2, **Ak, **Bk, **Aijk, **Bijk;
    double V, numA, numB;

    n = dims[0];
    dst = dims[1];

    /* critical to pass correct flag dst from R */
    Dx = alloc_matrix(n, n);
    Dy = alloc_matrix(n, n);
    if (dst) {
		vector2matrix(x, Dx, n, n, 1);
		vector2matrix(y, Dy, n, n, 1);
	}

    Dx2 = alloc_matrix(n, n);
    Dy2 = alloc_matrix(n, n);
    for (i = 0; i < n; i++)
        for (j = i; j < n; j++)
        {
            Dx2[i][j] = Dx[i][j] * Dx[i][j];
            Dy2[i][j] = Dy[i][j] * Dy[i][j];
            Dx2[j][i] = Dx2[i][j];
            Dy2[j][i] = Dy2[i][j];
        }

    for (l = 0; l < 4; l++)
        ACOV[l] = 0.0;

    for (k = 0; k < n; k++)
    {
        Aijk = alloc_matrix(n, n);
        Bijk = alloc_matrix(n, n);
        for (i = 0; i < n; i++)
        {   Aijk[i][i] = 0.0;
            Bijk[i][i] = 0.0;
            for ( j = i + 1; j < n; j++)
            {
                if (i == k || j == k)
                {
                    Aijk[i][j] = 0.0;
                    Bijk[i][j] = 0.0;
                }
                else
                {
                    numA = (Dx2[i][k] + Dx2[j][k] - Dx2[i][j]) / (2 * Dx[i][k] * Dx[j][k]);
                    numB = (Dy2[i][k] + Dy2[j][k] - Dy2[i][j]) / (2 * Dy[i][k] * Dy[j][k]);
                    if ((numA - 1) > -DBL_EPSILON)
                        numA = 1.0;
                    else if ((numA + 1) < DBL_EPSILON)
                        numA = -1.0;
                    else if (isnan(numA))
                        numA = 1.0;
                
                    if ((numB - 1) > -DBL_EPSILON)
                        numB = 1.0;
                    else if ((numB+ 1) < DBL_EPSILON)
                        numB = -1.0;
                    else if (isnan(numB))
                        numB = 1.0;
                /*Rprintf("%f", numA);*/

                    Aijk[i][j] = acos(numA);
                    Bijk[i][j] = acos(numB);
                }

                Aijk[j][i] = Aijk[i][j];
                Bijk[j][i] = Bijk[i][j];
            }
        }
        Ak = alloc_matrix(n, n);
        Bk = alloc_matrix(n, n);
        D_center(Aijk, Ak, n);
        D_center(Bijk, Bk, n);        
        free_matrix(Aijk, n, n);
        free_matrix(Bijk, n, n);
        
        for (i=0; i<n; i++)
            for (j=0; j<n; j++) {
                ACOV[0] += Ak[i][j]*Bk[i][j];
                ACOV[2] += Ak[i][j]*Ak[i][j];
                ACOV[3] += Bk[i][j]*Bk[i][j];
            }

        free_matrix(Ak, n, n);
        free_matrix(Bk, n, n);
    }
    
    free_matrix(Dx2, n, n);
    free_matrix(Dy2, n, n);

    n3 =  ((double) n ) * n * n;

    for (l=0; l<4; l++) {
        ACOV[l] /= n3;
        if (ACOV[l] > 0)
            ACOV[l] = sqrt(ACOV[l]);
            else ACOV[l] = 0.0;
    }
    
    /* compute aCor(x, y) */
    V = ACOV[2]*ACOV[3];
    if (V > DBL_EPSILON)
        ACOV[1] = ACOV[0] / sqrt(V);
        else ACOV[1] = 0.0;

    return;
}

void bcaCOV(double *x, double *y, int *dims, double *ACOV) {
    /*  computes aCov(x,y), aCor(x,y), aVar(x), aVar(y)
        V-statistic is n*aCov^2 
        x, y: two n by n distance matrixes
        dims[0] = n (sample size)
        dims[1] = dst (logical, TRUE if x, y are distances)
        ACOV  : vector [aCov, aCor, aVar(x), aVar(y), mean(A), mean(B)]
     */
    int    i, j, k, l, n, n3;
    int    dst;
    double **Dx, **Dy, **Dx2, **Dy2, **Ak, **Bk, **Aijk, **Bijk;
    double V, numA, numB;

    n = dims[0];
    dst = dims[1];

    /* critical to pass correct flag dst from R */
    Dx = alloc_matrix(n, n);
    Dy = alloc_matrix(n, n);
    if (dst) {
		vector2matrix(x, Dx, n, n, 1);
		vector2matrix(y, Dy, n, n, 1);
	}

    Dx2 = alloc_matrix(n, n);
    Dy2 = alloc_matrix(n, n);
    for (i = 0; i < n; i++)
        for (j = i; j < n; j++)
        {
            Dx2[i][j] = Dx[i][j] * Dx[i][j];
            Dy2[i][j] = Dy[i][j] * Dy[i][j];
            Dx2[j][i] = Dx2[i][j];
            Dy2[j][i] = Dy2[i][j];
        }

    for (l = 0; l < 4; l++)
        ACOV[l] = 0.0;

    for (k = 0; k < n; k++)
    {
        Aijk = alloc_matrix(n, n);
        Bijk = alloc_matrix(n, n);
        for (i = 0; i < n; i++)
        {   Aijk[i][i] = 0.0;
            Bijk[i][i] = 0.0;
            for ( j = i + 1; j < n; j++)
            {
                if (i == k || j == k)
                {
                    Aijk[i][j] = 0.0;
                    Bijk[i][j] = 0.0;
                }
                else
                {
                    numA = (Dx2[i][k] + Dx2[j][k] - Dx2[i][j]) / (2 * Dx[i][k] * Dx[j][k]);
                    numB = (Dy2[i][k] + Dy2[j][k] - Dy2[i][j]) / (2 * Dy[i][k] * Dy[j][k]);
                    if ((numA - 1) > -DBL_EPSILON)
                        numA = 1.0;
                    else if ((numA + 1) < DBL_EPSILON)
                        numA = -1.0;
                    else if (isnan(numA))
                        numA = 1.0;
                
                    if ((numB - 1) > -DBL_EPSILON)
                        numB = 1.0;
                    else if ((numB+ 1) < DBL_EPSILON)
                        numB = -1.0;
                    else if (isnan(numB))
                        numB = 1.0;
                /*Rprintf("%f", numA);*/

                    Aijk[i][j] = acos(numA);
                    Bijk[i][j] = acos(numB);
                }

                Aijk[j][i] = Aijk[i][j];
                Bijk[j][i] = Bijk[i][j];
            }
        }
        Ak = alloc_matrix(n, n);
        Bk = alloc_matrix(n, n);
        U_center(Aijk, Ak, n);
        U_center(Bijk, Bk, n);        
        free_matrix(Aijk, n, n);
        free_matrix(Bijk, n, n);
        
        for (i=0; i<n; i++)
            for (j=0; j<n; j++) {
                ACOV[0] += Ak[i][j]*Bk[i][j];
                ACOV[2] += Ak[i][j]*Ak[i][j];
                ACOV[3] += Bk[i][j]*Bk[i][j];
            }

        free_matrix(Ak, n, n);
        free_matrix(Bk, n, n);
    }
    
    free_matrix(Dx2, n, n);
    free_matrix(Dy2, n, n);

    n3 =  ((double) n ) * n * (n-3);

    for (l=0; l<4; l++) {
        ACOV[l] /= n3;
    }
    
    /* compute aCor(x, y) */
    V = ACOV[2]*ACOV[3];
    if (V > DBL_EPSILON)
        ACOV[1] = ACOV[0] / sqrt(V);
        else ACOV[1] = 0.0;

    return;
}



double D_center(double **akl, double **A, int n) {
    /*
    computes the double centered distance matrix for distance matrix akl
    a_{ij} - a_{i.}/n - a_{.j}/n + a_{..}/n^2, all i, j
    */
    int j, k;
    double *akbar;
    double abar;

    akbar = Calloc(n, double);
    abar = 0.0;
    for (k=0; k<n; k++) {
        akbar[k] = 0.0;
        for (j=0; j<n; j++) {
            akbar[k] += akl[k][j];
        }
        abar += akbar[k];
        akbar[k] /= (double) n;
    }
    abar /= (double) (n*n);

    for (k=0; k<n; k++)
        for (j=k; j<n; j++) {
            A[k][j] = akl[k][j] - akbar[k] - akbar[j] + abar;
            A[j][k] = A[k][j];
        }
    Free(akbar);
    return(abar);
}

double U_center(double **akl, double **A, int n) {
    /*
    computes the A_{kl}^U distances from the distance matrix (Dx_{kl}) for dCov^U
    U-centering: if Dx = (a_{ij}) then compute U-centered A^U using
    a_{ij} - a_{i.}/(n-2) - a_{.j}/(n-2) + a_{..}/((n-1)(n-2)), i \neq j
    and zero diagonal
    */
    int j, k;
    double *akbar;
    double abar;

    akbar = Calloc(n, double);
    abar = 0.0;
    for (k=0; k<n; k++) {
        akbar[k] = 0.0;
        for (j=0; j<n; j++) {
            akbar[k] += akl[k][j];
        }
        abar += akbar[k];
        akbar[k] /= (double) (n-2);
    }
    abar /= (double) ((n-1)*(n-2));

    for (k=0; k<n; k++)
        for (j=k; j<n; j++) {
            A[k][j] = akl[k][j] - akbar[k] - akbar[j] + abar;
            A[j][k] = A[k][j];
        }
    
    for (k=0; k<n; k++) {
        A[k][k] = 0.0;
    }

    Free(akbar);
    return(abar);
}



void acorsis(double *x, double *y, int *dims, double *acor_stat) 
{
    /*  sure independence screening using angle correlation
        x: n by d data matrix by row (use as.double(t(x)) in R if necessary)
        y: n by n distance matrix byrow
        dims[0] = n (sample size)
        dims[1] = p (number of features)
        acor_stat  : vector containing p acorr of features in order
    */
    int    i, j, k, l, n, n3, p;
    int    i_feature;
    double **Dx_temp, **Dy, **Dx_temp2, **Dy2, **Ak, **Bk, **Aijk, **Bijk, ***B;
    double *x_temp, *acov_temp;
    double V, numA, numB;

    n = dims[0];
    p = dims[1];

    // get angle matrix B for y
    Dy = alloc_matrix(n, n); 
	vector2matrix(y, Dy, n, n, 1);
    Dy2 = alloc_matrix(n, n);
    for (i = 0; i < n; i++)
        for (j = i; j < n; j++)
        {
            Dy2[i][j] = Dy[i][j] * Dy[i][j];
            Dy2[j][i] = Dy2[i][j];
        }

    B = alloc_array(n, n, n);
    for (k = 0; k < n; k++)
    {
        Bijk = alloc_matrix(n, n);
        for (i = 0; i < n; i++)
        {   Bijk[i][i] = 0.0;
            for ( j = i + 1; j < n; j++)
            {
                if (i == k || j == k)
                    Bijk[i][j] = 0.0;
                else
                {
                    numB = (Dy2[i][k] + Dy2[j][k] - Dy2[i][j]) / (2 * Dy[i][k] * Dy[j][k]);
                    if ((numB - 1) > -DBL_EPSILON)
                        numB = 1.0;
                    else if ((numB + 1) < DBL_EPSILON)
                        numB = -1.0;
                    else if (isnan(numB))
                        numB = 1.0;
                    Bijk[i][j] = acos(numB);
                }
                Bijk[j][i] = Bijk[i][j];
            }
        }
        Bk = alloc_matrix(n, n);
        D_center(Bijk, Bk, n);
        free_matrix(Bijk, n, n);
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                B[i][j][k] = Bk[i][j];
        free_matrix(Bk, n, n);
    }
    free_matrix(Dy, n, n);
    free_matrix(Dy2, n, n);       
    for (i_feature = 0; i_feature < p; i_feature++)
        {   
            acov_temp = Calloc(4, double);
            for (l = 0; l < 4; l++)
                acov_temp[l] = 0.0;
        
            // get i_feature-th feature
            x_temp = Calloc(n, double); 
            for (i = 0; i < n; i++)
            {
                x_temp[i] =  (*(x+i*p+i_feature));
            } 
            // distance matrix of i_feature-th feature
            Dx_temp = alloc_matrix(n, n); 
            Euclidean_distance(x_temp, Dx_temp, n, 1);    
            Free(x_temp);        
            // square distance matrix
            Dx_temp2 = alloc_matrix(n, n); 
            for (i = 0; i < n; i++)
                for (j = i; j < n; j++)
                {
                    Dx_temp2[i][j] = Dx_temp[i][j] * Dx_temp[i][j];
                    Dx_temp2[j][i] = Dx_temp2[i][j];
                }
            //Angle matrix of i_feature-th feature
            for (k = 0; k < n; k++)
            {
                Aijk = alloc_matrix(n, n);
                for (i = 0; i < n; i++)
                {   
                    Aijk[i][i] = 0.0;
                    for (j = i + 1; j < n; j++)
                    {
                        if (i == k || j == k)
                            Aijk[i][j] = 0.0;
                        else
                        {
                            numA = (Dx_temp2[i][k] + Dx_temp2[j][k] - Dx_temp2[i][j]) / (2 * Dx_temp[i][k] * Dx_temp[j][k]);
                            if ((numA - 1.0) > -DBL_EPSILON)
                                numA = 1.0;
                            else if ((numA + 1.0) < DBL_EPSILON)
                                numA = -1.0;
                            else if (isnan(numA))
                                numA = 1.0;
                            Aijk[i][j] = acos(numA);
                        }
                        Aijk[j][i] = Aijk[i][j];
                    }
                }
                Ak = alloc_matrix(n, n);
                D_center(Aijk, Ak, n);
                free_matrix(Aijk, n, n);
                for (i=0; i<n; i++)
                    for (j=0; j<n; j++) {
                        acov_temp[0] += Ak[i][j]*B[i][j][k];
                        acov_temp[2] += Ak[i][j]*Ak[i][j];
                        acov_temp[3] += B[i][j][k]*B[i][j][k];
                    }
                free_matrix(Ak, n, n);
            }
            free_matrix(Dx_temp, n, n);
            free_matrix(Dx_temp2, n, n);
            // get acor(Y, Xi)
            n3 =  ((double) n ) * n * n;
            for (l=0; l<4; l++) 
            {
                acov_temp[l] /= n3;
                if (acov_temp[l] > DBL_EPSILON)
                    acov_temp[l] = sqrt(acov_temp[l]);
                    else acov_temp[l] = 0.0;
            }
            V = acov_temp[2]*acov_temp[3];
            if (V > DBL_EPSILON)
                acov_temp[1] = acov_temp[0] / sqrt(V);
            else acov_temp[1] = 0.0;

            acor_stat[i_feature] = acov_temp[1];
            Free(acov_temp);
        }
    free_array(B, n, n, n);
    return;
}

/*
void acor_sis(double *x, double *y, int *dims, double *acor_stat, int *nthread) {
#ifdef Angle_OMP_H_
    omp_set_dynamic(0);
    if (*nthread <= 0) {
        omp_set_num_threads(omp_get_num_procs());
    } else {
        omp_set_num_threads(*nthread);
    }
#endif
    acorsis(x, y, dims, acor_stat);
}
*/



/*
utilities functions
alloc_matrix, alloc_int_matrix, free_matrix, free_int_matrix:
     use R (Calloc, Free) instead of C (calloc, free) for memory management

   vector2matrix      copies double* arg into double** arg
   matrix_power       computes matrix power
   distance           computes Euclidean distance matrix from double**
   Euclidean_distance computes Euclidean distance matrix from double*

*/
double **alloc_matrix(int r, int c)
{
    /* allocate a matrix with r rows and c columns */
    int i;
    double **matrix;
    matrix = Calloc(r, double *);
    for (i = 0; i < r; i++)
    matrix[i] = Calloc(c, double);
    return matrix;
}


double ***alloc_array(int r, int c, int d)
{
    int i, j;
    double ***array;
    array = Calloc(r, double **);
    for (i = 0; i < r; i++)
        array[i] = Calloc(c, double*);
    for (i = 0; i < r; i++)
        for (j = 0; j < c; j++)
            array[i][j] = Calloc(d, double);
    return array;
}



int **alloc_int_matrix(int r, int c)
{
    /* allocate an integer matrix with r rows and c columns */
    int i;
    int **matrix;
    matrix = Calloc(r, int *);
    for (i = 0; i < r; i++)
    matrix[i] = Calloc(c, int);
    return matrix;
}

void free_matrix(double **matrix, int r, int c)
{
    /* free a matrix with r rows and c columns */
    int i;
    for (i = 0; i < r; i++) Free(matrix[i]);
    Free(matrix);
}

void free_array(double ***array, int r, int c, int d)
{
    /* free a matrix with r rows and c columns */
    int i, j;
    for (i = 0; i < r; i++) 
    {
        for (j = 0; j < c; j++)
            Free(array[i][j]);
        Free(array[i]);
    }
    Free(array);
}

void free_int_matrix(int **matrix, int r, int c)
{
    /* free an integer matrix with r rows and c columns */
    int i;
    for (i = 0; i < r; i++) Free(matrix[i]);
    Free(matrix);
}

void vector2matrix(double *x, double **y, int N, int d, int isroworder) {
    /* copy a d-variate sample into a matrix, N samples in rows */
    int i, k;
    if (isroworder == TRUE) {
        for (k=0; k<d; k++)
            for (i=0; i<N; i++)
                y[i][k] = (*(x+i*d+k));
        }
    else {
        for (k=0; k<N; k++)
            for (i=0; i<d; i++)
                y[i][k] = (*(x+k*N+i));
        }
    return;
}

double **matrix_power(double **Dx, int n, double index)
{
    /*
        Dx is an n by n Euclidean distance matrix
        if index NEQ 1, compute D^index
    */
    int i, j;
    double **Dx_new;

    Dx_new = alloc_matrix(n, n);

    if (fabs(index - 1) > DBL_EPSILON) {
        for (i=0; i<n; i++)
            for (j=i+1; j<n; j++) {
                Dx_new[i][j] = R_pow(Dx[i][j], index);
                Dx_new[j][i] = Dx_new[i][j];
            }
    }
    return Dx_new;
}


void distance(double **data, double **D, int N, int d) {
    /*
       compute the distance matrix of sample in N by d matrix data
       equivalent R code is:  D <- as.matrix(dist(data))
    */
    int    i, j, k;
    double dif;
    for (i=0; i<N; i++) {
        D[i][i] = 0.0;
        for (j=i+1; j<N; j++) {
            D[i][j] = 0.0;
            for (k=0; k<d; k++) {
                dif = data[i][k] - data[j][k];
                D[i][j] += dif*dif;
            }
            D[i][j] = sqrt(D[i][j]);
            D[j][i] = D[i][j];
        }
    }
    return;
}


void Euclidean_distance(double *x, double **Dx, int n, int d)
{
    /*
        interpret x as an n by d matrix, in row order (n vectors in R^d)
        compute the Euclidean distance matrix Dx
    */
    int i, j, k, p, q;
    double dsum, dif;
    for (i=1; i<n; i++) {
        Dx[i][i] = 0.0;
        p = i*d;
        for (j=0; j<i; j++) {
            dsum = 0.0;
            q = j*d;
            for (k=0; k<d; k++) {
                dif = *(x+p+k) - *(x+q+k);
                dsum += dif*dif;
            }
            Dx[i][j] = Dx[j][i] = sqrt(dsum);
        }
    }
}

/*
 Apply quicksort algorithm to a, and the index exchange result is recorded in idx.
 array: an array to be sort in ascending order
 idx: an order index array. idx[i] = j means the i smallest value of array is a[j].
*/
void quicksort(double *array, int *idx, int l, int u) {
    int i, m, idx_temp;
    double a_temp;
    if (l >= u)
        return;
    m = l;
    for (i = l + 1; i <= u; i++) {
        if (array[i] < array[l]) {
            ++m;
            idx_temp = idx[m];
            idx[m] = idx[i];
            idx[i] = idx_temp;

            a_temp = array[m];
            array[m] = array[i];
            array[i] = a_temp;
        }
    }
    idx_temp = idx[l];
    idx[l] = idx[m];
    idx[m] = idx_temp;

    a_temp = array[l];
    array[l] = array[m];
    array[m] = a_temp;

    quicksort(array, idx, l, m - 1);
    quicksort(array, idx, m + 1, u);
}

