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
    
    static bool firtTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    static int num_inputs = 0;
    if (firtTime) {
        
         tensor *t = (tensor *)nn->data->training->set;
        
        if (nn->is_dense_network) {
            num_inputs = t->shape[0][0][0] / nn->dense->parameters->topology[0];
        } else if(nn->is_conv2d_network) {
            num_inputs = t->shape[0][0][0];
        }
        
        firtTime = false;
    }
    
    float sgn;
    if (weights[offset+(stride1+(stride2+((i*n)+j)))] > 0) {
        sgn = 1.0f;
    } else if (weights[offset+(stride1+(stride2+((i*n)+j)))] < 0) {
        sgn = -1.0f;
    } else {
        sgn = 0.0;
    }
    return weights[offset+(stride1+(stride2+((i*n)+j)))] -
    ((eta*lambda)/(float)num_inputs)*sgn;
}

float l2_regularizer(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2) {
    
    static bool firtTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    static int num_inputs = 0;
    if (firtTime) {
        
        tensor *t = (tensor *)nn->data->training->set;
        
        if (nn->is_dense_network) {
            num_inputs = t->shape[0][0][0] / nn->dense->parameters->topology[0];
        } else if(nn->is_conv2d_network) {
            num_inputs = t->shape[0][0][0];
        }
        
        firtTime = false;
    }
    
    return (1.0f-((eta*lambda)/(float)num_inputs))*weights[offset+(stride1+(stride2+((i*n)+j)))];
}
