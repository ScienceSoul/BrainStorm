//
//  Regularization.c
//  BrainStorm
//
//  Created by Hakime Seddik on 12/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#include <stdio.h>
#include "Regularization.h"
#include "NeuralNetwork.h"

float l0_regularizer(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2) {
    return weights[offset+(stride1+(stride2+((i*n)+j)))];
}

float l1_regularizer(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    float sgn;
    if (weights[offset+(stride1+(stride2+((i*n)+j)))] > 0) {
        sgn = 1.0f;
    } else if (weights[offset+(stride1+(stride2+((i*n)+j)))] < 0) {
        sgn = -1.0f;
    } else {
        sgn = 0.0;
    }
    return weights[offset+(stride1+(stride2+((i*n)+j)))] -
    ((eta*lambda)/(float)nn->data->training->m)*sgn;
}

float l2_regularizer(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    return (1.0f-((eta*lambda)/(float)nn->data->training->m))*weights[offset+(stride1+(stride2+((i*n)+j)))];
}
