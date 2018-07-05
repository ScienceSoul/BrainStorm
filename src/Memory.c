//
//  memory.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#include <stddef.h>
#include <stdarg.h>

#include "Memory.h"
#include "Utils.h"

#define FI_END 1
#define FREE_ARG void*

int * _Nonnull intvec(long nl, long nh)
{
    int *v;
    
    v = (int *)malloc((size_t) ((nh-nl+1+FI_END)*sizeof(int)));
    if (!v) fatal("Memory allocation failure for the integer vector.");
    return v-nl+FI_END;
}

float * _Nonnull floatvec(long nl, long nh) {
    float *v;
    
    v = (float *)malloc((size_t) ((nh-nl+1+FI_END)*sizeof(float)));
    if (!v) fatal("Memory allocation failure for the float vector.");
    return v-nl+FI_END;
}

float * _Nonnull * _Nonnull floatmatrix(long nrl, long nrh, long ncl, long nch)
{
    long i, nrow=nrh-nrl+1, ncol=nch-ncl+1;
    float **m;
    
    m = (float **) malloc((size_t) ((nrow+FI_END)*sizeof(float*)));
    if (!m) fatal("Memory allocation failure for the float matrix 1.");
    m += FI_END;
    m -= nrl;
    
    m[nrl]=(float *) malloc((size_t) ((nrow*ncol+FI_END)*sizeof(float)));
    if (!m[nrl]) fatal("Memory allocation failure for the float matrix 2.");
    m[nrl] += FI_END;
    m[nrl] -= ncl;
    
    for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
    
    return m;
}

void free_ivector(int * _Nonnull v, long nl, long nh)
{
    free((FREE_ARG) (v+nl-FI_END));
}

void free_fvector(float * _Nonnull v, long nl, long nh)
{
    free((FREE_ARG) (v+nl-FI_END));
}

void free_fmatrix(float * _Nonnull * _Nonnull m, long nrl, long nrh, long ncl, long nch)
{
    free((FREE_ARG) (m[nrl]+ncl-FI_END));
    free((FREE_ARG) (m+nrl-FI_END));
}
