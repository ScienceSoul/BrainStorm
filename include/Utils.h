//
//  Utils.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#ifndef Utils_h
#define Utils_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>

#define DEFAULT_CONSOLE_WRITER "BrainStorm"

#define MAX_NUMBER_NETWORK_LAYERS 500
#define MAX_TENSOR_RANK 5
#define MAX_SUPPORTED_PARAMETERS 50
#define MAX_LONG_STRING_LENGTH 256
#define MAX_SHORT_STRING_LENGTH 128

void __attribute__((overloadable)) fatal(char head[_Nonnull]);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull]);
void __attribute__((overloadable)) fatal(char head [_Nonnull], char message[_Nonnull], char string [_Nonnull]);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull], int n);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull], double n);
void __attribute__((overloadable)) fatal(char head [_Nonnull], char message[_Nonnull], char string [_Nonnull]);

void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull]);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull], int n);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull], double n);

void __attribute__((overloadable)) shuffle(float * _Nonnull * _Nonnull array, unsigned int len1, unsigned int len2);
void __attribute__((overloadable)) shuffle(void * _Nonnull object);
void __attribute__((overloadable))shuffle(void * _Nonnull neural, void * _Nonnull object, void * _Nullable associate, int num_classifications);

void __attribute__((overloadable))parseArgument(const char * _Nonnull argument, const char * _Nonnull argumentName, int  * _Nonnull result, unsigned int * _Nonnull numberOfItems, unsigned int * _Nonnull len);
void __attribute__((overloadable)) parseArgument(const char * _Nonnull argument, const char * _Nonnull argumentName, char result[_Nonnull][128], unsigned int * _Nonnull numberOfItems, unsigned int * _Nonnull len);
void __attribute__ ((overloadable))parseArgument(const char * _Nonnull argument, const char * _Nonnull argumentName, float * _Nonnull result, unsigned int * _Nonnull numberOfItems, unsigned int *_Nonnull len);


float randn(float mu, float sigma);
float random_uniform(float r1, float r2);

int __attribute__((overloadable)) max(int x, int y);
int __attribute__((overloadable)) max(int x, int y, int z);
int __attribute__((overloadable)) max(int w, int x, int y, int z);
float __attribute__((overloadable)) max(float x, float y);
float __attribute__((overloadable)) max(float x, float y, float z);

int __attribute__((overloadable)) min(int x, int y);
int __attribute__((overloadable)) min(int x, int y, int z);
int __attribute__((overloadable)) min(int w, int x, int y, int z);
float __attribute__((overloadable)) min(float x, float y);
float __attribute__((overloadable)) min(float x, float y, float z);

int __attribute__((overloadable)) minv(int * _Nonnull a, unsigned int num_elements);
float __attribute__((overloadable)) minv(float * _Nonnull a, unsigned int num_elements);

int __attribute__((overloadable)) maxv(int * _Nonnull a, unsigned int num_elements);
float __attribute__((overloadable)) maxv(float * _Nonnull a, unsigned int num_elements);

float __attribute__((overloadable)) meanv(float * _Nonnull a, unsigned int num_elements);

float __attribute__((overloadable)) sve(float * _Nonnull a, unsigned int num_elements);

int __attribute__((overloadable)) argmax(int * _Nonnull a, unsigned int num_elements);
int __attribute__((overloadable)) argmax(float * _Nonnull a, unsigned int num_elements);

float sigmoid(float z, float * _Nullable vec, unsigned int * _Nullable n);
float sigmoidPrime(float z);

float tan_h(float z, float * _Nullable vec, unsigned int * _Nullable n);
float tanhPrime(float z);

float relu(float z, float * _Nullable vec, unsigned int * _Nullable n);
float reluPrime(float z);

float leakyrelu(float z, float * _Nullable vec, unsigned int * _Nullable n);
float leakyreluPrime(float z);

float elu(float z, float * _Nullable vec, unsigned int * _Nullable n);
float eluPrime(float z);

float softmax(float z, float * _Nullable vec, unsigned int * _Nullable n);

float crossEntropyCost(float * _Nonnull a, float * _Nonnull y, unsigned int n);

float __attribute__((overloadable)) frobeniusNorm(float * _Nonnull * _Nonnull mat, unsigned int m, unsigned int n);
float __attribute__((overloadable)) frobeniusNorm(float * _Nonnull mat, unsigned int n);

void  __attribute__((overloadable)) nanToNum(float * _Nonnull array, unsigned int n);

int nearestPower2(int num);

void  __attribute__((overloadable)) shape(unsigned int dest[_Nonnull][MAX_TENSOR_RANK][1], unsigned int layers, unsigned int rank, int * _Nonnull vector);

void  __attribute__((overloadable)) shape(unsigned int dest[_Nonnull][MAX_TENSOR_RANK][1], unsigned int rank, int * _Nonnull vector, unsigned int layer);

void __attribute__((overloadable)) swap(float * _Nonnull A, int i, int j, int k, int lda);
void __attribute__((overloadable)) swap(float * _Nonnull A, int i, int j, int lda);
void reverse_rows(float * _Nonnull A, int m, int n);
void transpose(float * _Nonnull A, int m, int n);

#endif /* Utils_h */
