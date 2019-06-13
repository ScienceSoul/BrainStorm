//
//  Utils.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#endif

#ifdef __linux__
    #include <bsd/stdlib.h>
#endif

#include "Utils.h"
#include "Memory.h"
#include "NetworkUtils.h"

static int formatType;
void format(char * _Nullable head, char * _Nullable message, int * _Nullable iValue, double * _Nullable dValue, char * _Nullable str);

void __attribute__((overloadable)) fatal(char head[]) {
    
    formatType = 1;
    format(head, NULL, NULL, NULL, NULL);
}

void __attribute__((overloadable)) fatal(char head[], char message[]) {
    
    formatType = 2;
    format(head, message, NULL, NULL, NULL);
}

void __attribute__((overloadable)) fatal(char head[], char message[], int n) {
    
    formatType = 3;
    format(head, message, &n, NULL, NULL);
}

void __attribute__((overloadable)) fatal(char head[], char message[], double n) {
    
    formatType = 4;
    format(head, message, NULL, &n, NULL);
}

void __attribute__((overloadable)) fatal(char head [_Nonnull], char message[_Nonnull], char str[_Nonnull]) {
    formatType = 5;
    format(head, message, NULL, NULL, str);
}


void __attribute__((overloadable)) warning(char head[], char message[])
{
    fprintf(stdout, "%s: %s\n", head, message);
}

void __attribute__((overloadable)) warning(char head[], char message[], int n)
{
    fprintf(stdout, "%s: %s %d\n", head, message, n);
}

void __attribute__((overloadable)) warning(char head[], char message[], double n)
{
    fprintf(stdout, "%s: %s %f\n", head, message, n);
}

void format(char * _Nullable head, char * _Nullable message, int * _Nullable iValue, double * _Nullable dValue, char * _Nullable str) {
    
    fprintf(stderr, "##                    A FATAL ERROR occured                   ##\n");
    fprintf(stderr, "##        Please look at the error log for diagnostic         ##\n");
    fprintf(stderr, "\n");
    if (formatType == 1) {
        fprintf(stderr, "%s: Program will abort...\n", head);
    } else if (formatType == 2) {
        fprintf(stderr, "%s: %s\n", head, message);
    } else if (formatType == 3) {
        fprintf(stderr, "%s: %s %d.\n", head, message, *iValue);
    } else if (formatType == 4) {
        fprintf(stderr, "%s: %s %f.\n", head, message, *dValue);
    } else if (formatType == 5) {
        fprintf(stderr, "%s: %s %s.\n", head, message, str);
    }
    if (formatType == 2 || formatType == 3 || formatType == 4 || formatType == 5)
        fprintf(stderr, "Program will abort...\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "################################################################\n");
    fprintf(stderr, "################################################################\n");
    exit(EXIT_FAILURE);
}

// ---------------------------------------
// Shuffle a 2D array of the form arr[][]
// ---------------------------------------
void __attribute__((overloadable)) shuffle(float * _Nonnull * _Nonnull array, unsigned int len1, unsigned int len2) {
    
    float t[len2];
    
    if (len1 > 1)
    {
        for (int i = 0; i < len1 - 1; i++)
        {
            int j = i + rand() / (RAND_MAX / (len1 - i) + 1);
            for (int k=0; k<len2; k++) {
                t[k] = array[j][k];
            }
            for (int k=0; k<len2; k++) {
                array[j][k] = array[i][k];
            }
            for (int k=0; k<len2; k++) {
                array[i][k] = t[k];
            }
        }
    }
}

// --------------------------
// Shuffle a 4D input tensor
// --------------------------
void __attribute__((overloadable))shuffle(void * _Nonnull object) {
    
    tensor *input = (tensor *)object;
    
    unsigned int dim = 1;
    for (int i=1; i<input->rank; i++) {
        dim = dim * input->shape[0][i][0];
    }
    float t1[dim];
    
    if (input->shape[0][0][0] > 1) {
        int stride1 = 0;
        for (int i=0; i<input->shape[0][0][0]-1; i++) {
            int j = i + rand() / (RAND_MAX / (input->shape[0][0][0] - i) + 1);
            int jump1 = max((j-1),0) * dim;
            for (int k=0; k<dim; k++) {
                t1[k] = input->val[jump1+k];
            }
            for (int k=0; k<dim; k++) {
                input->val[jump1+k] = input->val[stride1+k];
            }
            for (int k=0; k<dim; k++) {
                input->val[stride1+k] = t1[k];
            }
            stride1 = stride1 + dim;
        }
    }
}

void __attribute__((overloadable))parse_argument(const char * _Nonnull argument, const char * _Nonnull argumentName, int  * _Nonnull result, unsigned int * _Nonnull numberOfItems, unsigned int * _Nonnull len) {
    int idx = 0;
    *numberOfItems = 0;
    
    fprintf(stdout, "%s: parsing the key value <%s>: %s.\n", DEFAULT_CONSOLE_WRITER, argumentName, argument);
    
    size_t length = strlen(argument);
    if (argument[0] != '[' || argument[length-1] != ']') fatal(DEFAULT_CONSOLE_WRITER, "syntax error in key value. Collections must use the [ ] syntax.");
    if ( argument[length-2] == ')') fatal(DEFAULT_CONSOLE_WRITER, "a range definition can't be used for the network output layer.");
    
    unsigned int checkRange = 0;
    while (1) {
        if (argument[idx] == '[') {
            if (argument[idx+1] == ',' || argument[idx+1] == '[') fatal(DEFAULT_CONSOLE_WRITER, "syntax error possibly <[,> or <[[> in key value");
            if (argument[idx+1] == '(') fatal(DEFAULT_CONSOLE_WRITER, "a range definition can't be used for the network input layer");
            idx++;
        }
        
        if (argument[idx] == ']') {
            (*numberOfItems)++;
            break;
        } else if (argument[idx] == '(') { // Begining of the definition of a range
            int layerNumbering[2];
            memset(layerNumbering, 0, sizeof(layerNumbering));
            int count = 0;
            idx++;
            while (1) {
                if (argument[idx] == '~') {
                    idx++;
                    count++;
                } else if (argument[idx] == ';') { // The number of units this range of the network will have
                    int numberOfUnits = 0;
                    idx++;
                    while (1) {
                        if (argument[idx] == ')') {
                            if (argument[idx+1] != ',') fatal(DEFAULT_CONSOLE_WRITER, "syntax error in range definition. <}> should be followed by <,>");
                            if ((layerNumbering[0]-1) - checkRange != 1) fatal(DEFAULT_CONSOLE_WRITER, "range definition is not compatible with a correct topology of the network.");
                            for (int i=layerNumbering[0]-1; i<layerNumbering[1]; i++) {
                                if (*numberOfItems >= *len) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when parsing the key:", (char *)argumentName);
                                result[*numberOfItems] = numberOfUnits;
                                (*numberOfItems)++;
                            }
                            (*numberOfItems)--; // We are going one step ahead from the previous loop, so come back one step back
                            checkRange = layerNumbering[1] - 1;
                            break;
                        } else {
                            int digit = argument[idx] - '0';
                            if (digit < 0 || digit > 9) fatal(DEFAULT_CONSOLE_WRITER, "NaN in key value.");
                            numberOfUnits = numberOfUnits * 10 + digit;
                            idx++;
                        }
                    }
                    idx++;
                    break;
                } else {
                    int digit = argument[idx] - '0';
                    if (digit < 0 || digit > 9) fatal(DEFAULT_CONSOLE_WRITER, "NaN in key value.");
                    layerNumbering[count] = layerNumbering[count] * 10 + digit;
                    idx++;
                }
            }
        } else if (argument[idx] == ',') {
            if (argument[idx+1] == ']' || argument[idx+1] == ',') fatal(DEFAULT_CONSOLE_WRITER, "syntax error possibly <,]> or <,,> in key value.");
            (*numberOfItems)++;
            idx++;
        } else {
            int digit = argument[idx] - '0';
            if (digit < 0 || digit > 9) fatal(DEFAULT_CONSOLE_WRITER, "NaN in key value.");
            if (*numberOfItems >= *len) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when parsing the key:", (char *)argumentName);
            result[*numberOfItems] = result[*numberOfItems] * 10 + digit;
            idx++;
        }
    }
}

void __attribute__((overloadable)) parse_argument(const char * _Nonnull argument, const char * _Nonnull argumentName, char result[_Nonnull][128], unsigned int * _Nonnull numberOfItems, unsigned int * _Nonnull len) {
    
    int idx = 0;
    int bf_idx = 0;
    *numberOfItems = 0;
    char buffer[MAX_SHORT_STRING_LENGTH];
    
    
    fprintf(stdout, "%s: parsing the key value <%s>: %s.\n", DEFAULT_CONSOLE_WRITER, argumentName, argument);
    
    size_t length = strlen(argument);
    if (argument[0] != '[' || argument[length-1] != ']') fatal(DEFAULT_CONSOLE_WRITER, "syntax error in key value. Collections must use the [ ] syntax.");
    if ( argument[length-2] == ')') fatal(DEFAULT_CONSOLE_WRITER, "a range definition can't be used to define the activation function at the network output layer.");
    
    unsigned int checkRange = 0;
    memset(buffer, 0, sizeof(buffer));
    while (1) {
        if (argument[idx] == '[') {
            if (argument[idx+1] == ',' || argument[idx+1] == '[') fatal(DEFAULT_CONSOLE_WRITER, "syntax error possibly <[,> or <[[> in key value");
            idx++;
        }
        if (argument[idx] == '~') {
            memset(result[*numberOfItems], 0, sizeof(result[*numberOfItems]));
            memcpy(result[*numberOfItems], buffer, strlen(buffer));
            (*numberOfItems)++;
            break;
        } else if (argument[idx] == '(') { // Begining of the definition of a range
            int layerNumbering[2];
            memset(buffer, 0, sizeof(buffer));
            memset(layerNumbering, 0, sizeof(layerNumbering));
            int count = 0;
            bf_idx = 0;
            idx++;
            while (1) {
                if (argument[idx] == '~') {
                    idx++;
                    count++;
                } else if (argument[idx] == ';') { // The activatiuon functions this range of the network will use
                    idx++;
                    while (1) {
                        if (argument[idx] == ')') {
                            if (argument[idx+1] != ',') fatal(DEFAULT_CONSOLE_WRITER, "syntax error in range definition. <}> should be followed by <,>");
                            if ((layerNumbering[0]-1) - checkRange != 1) fatal(DEFAULT_CONSOLE_WRITER, "range definition is not compatible with a correct topology of the network.");
                            for (int i=layerNumbering[0]-1; i<layerNumbering[1]; i++) {
                                if (*numberOfItems >= *len) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when parsing the key:", (char *)argumentName);
                                memset(result[*numberOfItems], 0, sizeof(result[*numberOfItems]));
                                memcpy(result[*numberOfItems], buffer, strlen(buffer));
                                (*numberOfItems)++;
                            }
                            checkRange = layerNumbering[1] - 1;
                            break;
                        } else {
                            if (bf_idx >= MAX_SHORT_STRING_LENGTH) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when parsing the key:", (char *)argumentName);
                            buffer[bf_idx] = argument[idx];
                            bf_idx++;
                            idx++;
                        }
                    }
                    idx++;
                    memset(buffer, 0, sizeof(buffer));
                    bf_idx = 0;
                    break;
                } else {
                    int digit = argument[idx] - '0';
                    layerNumbering[count] = layerNumbering[count] * 10 + digit;
                    idx++;
                }
            }
        } else if (argument[idx] == ',' || argument[idx] == ']') {
            if (argument[idx] == ',') {
                if (argument[idx+1] == ']' || argument[idx+1] == ',') fatal(DEFAULT_CONSOLE_WRITER, "syntax error possibly <,]> or <,,> in key value.");
            }
            
            if (argument[idx-1] == ')') { // We got here from a previous range definition so jump to next iteration
                idx++;
                continue;
            }
            
            if (*numberOfItems >= *len) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when parsing the key:", (char *)argumentName);
            memset(result[*numberOfItems], 0, sizeof(result[*numberOfItems]));
            memcpy(result[*numberOfItems], buffer, strlen(buffer));
            (*numberOfItems)++;
            if (argument[idx] == ']') break;
            idx++;
            memset(buffer, 0, sizeof(buffer));
            bf_idx = 0;
            checkRange++;
        } else {
            if (bf_idx >= MAX_SHORT_STRING_LENGTH) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when parsing the key:", (char *)argumentName);
            buffer[bf_idx] = argument[idx];
            bf_idx++;
            idx++;
        }
    }
}

void __attribute__ ((overloadable))parse_argument(const char * _Nonnull argument, const char * _Nonnull argumentName, float * _Nonnull result, unsigned int * _Nonnull numberOfItems, unsigned int *_Nonnull len) {
    
    int idx = 0;
    int bf_idx = 0;
    *numberOfItems = 0;
    char buffer[MAX_SHORT_STRING_LENGTH];
    
    
    fprintf(stdout, "%s: parsing the key value <%s>: %s.\n", DEFAULT_CONSOLE_WRITER, argumentName, argument);
    
    size_t lenght = strlen(argument);
    if (argument[0] != '[' || argument[lenght-1] != ']') fatal(DEFAULT_CONSOLE_WRITER, "syntax error in key value. Collections must use the [ ] syntax.");
    if ( argument[lenght-2] == ')') fatal(DEFAULT_CONSOLE_WRITER, "a range definition can't be used to define the activation function at the network output layer.");
    
    memset(buffer, 0, sizeof(buffer));
    while (1) {
        if (argument[idx] == '[') {
            if (argument[idx+1] == ',' || argument[idx+1] == '[') fatal(DEFAULT_CONSOLE_WRITER, "syntax error possibly <[,> or <[[> in key value");
            idx++;
        }
        if (argument[idx] == ',' || argument[idx] == ']') {
            if (argument[idx] == ',') {
                if (argument[idx+1] == ']' || argument[idx+1] == ',') fatal(DEFAULT_CONSOLE_WRITER, "syntax error possibly <,]> or <,,> in key value.");
            }
            if (*numberOfItems >= *len) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when parsing the key:", (char *)argumentName);
            result[*numberOfItems] = strtof(buffer, NULL);
            (*numberOfItems)++;
            if (argument[idx] == ']') break;
            idx++;
            memset(buffer, 0, sizeof(buffer));
            bf_idx = 0;
        } else {
            if (bf_idx >= MAX_SHORT_STRING_LENGTH) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when parsing the key:", (char *)argumentName);
            buffer[bf_idx] = argument[idx];
            bf_idx++;
            idx++;
        }
    }
}


// Generate random numbers from Normal Distribution (Gauss Distribution) with mean mu and standard deviation sigma
// using the Marsaglia and Bray method
float randn(float mu, float sigma) {
    
    float U1, U2, W, mult;
    static float X1, X2;
    static int call = 0;
    
    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (float) X2);
    }
    
    do
    {
        U1 = -1 + ((float) rand () / RAND_MAX) * 2;
        U2 = -1 + ((float) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);
    
    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    
    call = !call;
    
    return (mu + sigma * (float) X1);
}

float random_uniform(float r1, float r2) {
    
    return ( ((float)arc4random()/0x100000000)*(r2-r1)+r1 );
}

int __attribute__((overloadable)) max(int x, int y) {
    
    return (x > y) ? x : y;
}

int __attribute__((overloadable)) max(int x, int y, int z) {
    
    int m = (x > y) ? x : y;
    return (z > m) ? z : m;
}

int __attribute__((overloadable)) max(int w, int x, int y, int z) {
    
    int m = (w > x) ? w : x;
    int n = (y > m) ? y : m;
    return (z > n) ? z : n;
}

float __attribute__((overloadable)) max(float x, float y) {
    
    return (x > y) ? x : y;
}

float __attribute__((overloadable)) max(float x, float y, float z) {
    
    float m = (x > y) ? x : y;
    return (z > m) ? z : m;
}

int __attribute__((overloadable)) min(int x, int y) {
    
    return (x < y) ? x : y;
}

int __attribute__((overloadable)) min(int x, int y, int z) {
    
    int m = (x < y) ? x : y;
    return (z < m) ? z : m;
}

int __attribute__((overloadable)) min(int w, int x, int y, int z) {
    
    int m = (w < x) ? w : x;
    int n = (y < m) ? y : m;
    return (z < n) ? z : n;
}

float __attribute__((overloadable)) min(float x, float y) {
    
    return (x < y) ? x : y;
}

float __attribute__((overloadable)) min(float x, float y, float z) {
    
    float m = (x < y) ? x : y;
    return (z < m) ? z : m;
}

int __attribute__((overloadable)) minv(int * _Nonnull a, unsigned int num_elements) {
    
    int min = INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] < min) {
            min = a[i];
        }
    }
    
    return min;
}

float __attribute__((overloadable)) minv(float * _Nonnull a, unsigned int num_elements) {
    
    float min = HUGE_VALF;
    for (int i=0; i<num_elements; i++) {
        if (a[i] < min) {
            min = a[i];
        }
    }
    
    return min;
}

int __attribute__((overloadable)) maxv(int * _Nonnull a, unsigned int num_elements)
{
    int max = -INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
        }
    }
    
    return max;
}

float __attribute__((overloadable)) maxv(float * _Nonnull a, unsigned int num_elements) {
    
    float max = -HUGE_VALF;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
        }
    }
    
    return max;
}

float __attribute__((overloadable)) meanv(float * _Nonnull a, unsigned int num_elements) {
    
    float sum = 0.0f;
    for (int i=0; i<num_elements; i++) {
        sum = sum + a[i];
    }
    
    return sum / num_elements;
}

float __attribute__((overloadable)) sve(float * _Nonnull a, unsigned int num_elements) {
    
    float sum = 0.0f;
    for (int i=0; i<num_elements; i++) {
        sum = sum + a[i];
    }
    
    return sum;
}

int __attribute__((overloadable)) argmax(int * _Nonnull a, unsigned int num_elements) {
    
    int idx=0, max = -INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
            idx = i;
        }
    }
    
    return idx;
}

int __attribute__((overloadable)) argmax(float * _Nonnull a, unsigned int num_elements) {
    
    int idx=0;
    float max = -HUGE_VALF;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
            idx = i;
        }
    }
    
    return idx;
}

//  The sigmoid fonction
float sigmoid(float z, float * _Nullable vec, unsigned int * _Nullable n) {
    return 1.0f / (1.0f + expf(-z));
}

// Derivative of the sigmoid function
float sigmoid_prime(float z) {
    return sigmoid(z,NULL,NULL) * (1.0f - sigmoid(z,NULL,NULL));
}

// The tanh function
float tan_h(float z, float * _Nullable vec, unsigned int * _Nullable n) {
    return tanhf(z);
}

// Derivative of the tanh function
float tanh_prime(float z) {
    float th = tanhf(z);
    return 1.0f - (th * th);
}

// The ReLU function
float relu(float z, float * _Nullable vec, unsigned int * _Nullable n) {
    return fmaxf(0.0f,z);
}

// Derivative of the ReLU function
float relu_prime(float z) {
    return (z < 0) ? 0.0f : 1.0f;
}

// The LeakyReLU function
float leakyrelu(float z, float * _Nullable vec, unsigned int * _Nullable n) {
    return (z < 0) ? 0.01f*z : z;
}

// Derivative of the LeakyReLU function
float leakyrelu_prime(float z) {
    return (z < 0) ? 0.01f : 1.0f;
}

// The ELU (exponential linear unit) function
float elu(float z, float * _Nullable vec, unsigned int * _Nullable n) {
    float alpha = 1.0f;
    return (z < 0) ? alpha * (expf(z) - 1.0f) : z ;
}

// Derivative of the ELU function
float elu_prime (float z) {
    float alpha = 1.0f;
    return (z < 0) ? alpha * expf(z) : 1;
}

// The softplus function
float softplus(float z, float * _Nullable vec, unsigned int * _Nullable n) {
    return logf(1.0f + expf(z));
}

// Derivative of the softplus function
float softplus_prime(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// The softmax function
float softmax(float z, float * _Nullable vec, unsigned int * _Nullable n) {
    float sum = 0;
    for (unsigned int i=0; i<*n; i++) {
        sum = sum + expf(vec[i]);
    }
    return expf(z) / sum;
}

//
//  Compute the Frobenius norm of a m x n matrix
//
float __attribute__((overloadable)) frobenius_norm(float * _Nonnull * _Nonnull mat, unsigned int m, unsigned int n) {
    
    float norm = 0.0f;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            norm = norm + powf(mat[i][j], 2.0f);
        }
    }
    
    return sqrtf(norm);
}

//
//  Compute the Frobenius norm of a m x n serialized matrix
//
float __attribute__((overloadable)) frobenius_norm(float * _Nonnull mat, unsigned int n) {
    
    float norm = 0.0f;
    for (int i=0; i<n; i++) {
        norm = norm + powf(mat[i], 2.0f);
    }
    
    return norm;
}

float cross_entropy_cost(float * _Nonnull a, float * _Nonnull y, unsigned int n) {
    
    float cost = 0.0f;
    float buffer[n];
    
    for (int i=0; i<n; i++) {
        buffer[i] = -y[i]*logf(a[i]) - (1.0f-y[i])*logf(1.0-a[i]);
    }
    nan_to_num(buffer, n);
#ifdef __APPLE__
    vDSP_sve(buffer, 1, &cost, n);
#else
    for (int i=0; i<n; i++) {
        cost = cost + buffer[i];
    }
#endif
    
    return cost;
}

void  __attribute__((overloadable)) nan_to_num(float * _Nonnull array, unsigned int n) {
    
    for (int i=0; i<n; i++) {
        if (isnan(array[i]) != 0) array[i] = 0.0f;
        
        if (isinf(array[i]) != 0) {
            if (array[i] > 0) {
                array[i] = HUGE_VALF;
            } else if (array[i] < 0) {
                array[i] = -HUGE_VALF;
            }
        }
    }
}

//
// Find the nearest power of 2 for a number
//  - Parameter n: the number to find the nearest power 2 of.
//  - Returns: The nearest power 2 of num (30 -> 32, 200 -> 256).
//
inline int  nearest_power2(int num) {
    
    int n = (num > 0) ? num - 1 : 0;
    
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n += 1;
    
    return n;
}

void __attribute__((overloadable)) shape(unsigned int dest[_Nonnull][MAX_TENSOR_RANK][1], unsigned int layers, unsigned int rank, int * _Nonnull vector) {
    
    for (int l=0; l<layers; l++) {
        for (int i=0; i<rank; i++) {
            dest[l][i][0] = vector[i];
        }
    }
}

void  __attribute__((overloadable)) shape(unsigned int dest[_Nonnull][MAX_TENSOR_RANK][1], unsigned int rank, int * _Nonnull vector, unsigned int layer) {
    
    for (int i=0; i<rank; i++) {
        dest[layer][i][0] = vector[i];
    }
}

void __attribute__((overloadable)) swap(float * _Nonnull A, int i, int j, int k, int lda) {
    float temp = A[j*lda+i];
    A[j*lda+i] = A[k*lda+i];
    A[k*lda+i] = temp;
}

void __attribute__((overloadable)) swap(float * _Nonnull A, int i, int j, int lda) {
    float temp = A[i*lda+j];
    A[i*lda+j] = A[j*lda+i];
    A[j*lda+i] = temp;
}

void reverse_rows(float * _Nonnull A, int m, int n) {
    for (int i=0; i<n; i++) {
        for (int j=0, k=n-1; j<k; j++, k--) {
            swap(A, i, j, k, n);
        }
    }
}

void transpose(float * _Nonnull A, int m, int n) {
    for (int i=0; i<m; i++) {
        for (int j=i; j<n; j++) {
            swap(A, i, j, n);
        }
    }
}
