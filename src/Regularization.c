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

float l0_regularizer(void * _Nonnull neural, int i, int j, int n, int stride) {
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    return nn->dense->weights->val[stride+((i*n)+j)];
}

float l1_regularizer(void * _Nonnull neural, int i, int j, int n, int stride) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    float sgn;
    if (nn->dense->weights->val[stride+((i*n)+j)] > 0) {
        sgn = 1.0f;
    } else if (nn->dense->weights->val[stride+((i*n)+j)] < 0) {
        sgn = -1.0f;
    } else {
        sgn = 0.0;
    }
    return nn->dense->weights->val[stride+((i*n)+j)] -
    ((nn->dense->parameters->eta*nn->dense->parameters->lambda)/(float)nn->data->training->m)*sgn;
}

float l2_regularizer(void * _Nonnull neural, int i, int j, int n, int stride) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    return (1.0f-((nn->dense->parameters->eta*nn->dense->parameters->lambda)/(float)nn->data->training->m))*nn->dense->weights->val[stride+((i*n)+j)];
}
