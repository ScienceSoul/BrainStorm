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

#define PROGRAM_NAME "FeedforwardNT"

#define MAX_NUMBER_NETWORK_LAYERS 500
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

void shuffle(float * _Nonnull * _Nonnull array, unsigned int len1, unsigned int len2);

void __attribute__((overloadable))parseArgument(const char * _Nonnull argument, const char * _Nonnull argumentName, int  * _Nonnull result, unsigned int * _Nonnull numberOfItems, unsigned int * _Nonnull len);
void __attribute__((overloadable)) parseArgument(const char * _Nonnull argument, const char * _Nonnull argumentName, char result[_Nonnull][128], unsigned int * _Nonnull numberOfItems, unsigned int * _Nonnull len);
void __attribute__ ((overloadable))parseArgument(const char * _Nonnull argument, const char * _Nonnull argumentName, float * _Nonnull result, unsigned int * _Nonnull numberOfItems, unsigned int *_Nonnull len);


float randn(float mu, float sigma);

int __attribute__((overloadable)) min_array(int * _Nonnull a, unsigned int num_elements);
float __attribute__((overloadable)) min_array(float * _Nonnull a, unsigned int num_elements);

int __attribute__((overloadable)) max_array(int * _Nonnull a, unsigned int num_elements);
float __attribute__((overloadable)) max_array(float * _Nonnull a, unsigned int num_elements);

int __attribute__((overloadable)) argmax(int * _Nonnull a, unsigned int num_elements);
int __attribute__((overloadable)) argmax(float * _Nonnull a, unsigned int num_elements);

float sigmoid(float z, float * _Nullable vec, unsigned int * _Nullable n);
float sigmoidPrime(float z);

float tan_h(float z, float * _Nullable vec, unsigned int * _Nullable n);
float tanhPrime(float z);

float relu(float z, float * _Nullable vec, unsigned int * _Nullable n);
float reluPrime(float z);

float softmax(float z, float * _Nullable vec, unsigned int * _Nullable n);

float crossEntropyCost(float * _Nonnull a, float * _Nonnull y, unsigned int n);

float __attribute__((overloadable)) frobeniusNorm(float * _Nonnull * _Nonnull mat, unsigned int m, unsigned int n);
float __attribute__((overloadable)) frobeniusNorm(float * _Nonnull mat, unsigned int n);

void  __attribute__((overloadable)) nanToNum(float * _Nonnull array, unsigned int n);

int nearestPower2(int num);

#endif /* Utils_h */
