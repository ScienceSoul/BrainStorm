//
//  Optimizers.c
//  BrainStorm
//
//  Created by Hakime Seddik on 12/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#include <stdio.h>
#include "Optimizers.h"
#include "NeuralNetwork.h"
#include "DenseNetOps.h"

void (* _Nullable ptr_init_func)(void * _Nonnull neural) = NULL;
void (* _Nullable ptr_gradient_descent_update_func)(void * _Nonnull  neural, unsigned int batch_size) = NULL;
void (* _Nullable ptr_momentum_update_func)(void * _Nonnull  neural, unsigned int batch_size) = NULL;
void (* _Nullable ptr_ada_grad_update_func)(void * _Nonnull  neural, unsigned int batch_size) = NULL;
void (* _Nullable ptr_rms_prop_update_func)(void * _Nonnull  neural, unsigned int batch_size) = NULL;
void (* _Nullable ptr_adam_update_func)(void * _Nonnull neural, unsigned int batch_size) = NULL;

void init_in_dense_net(void * _Nonnull neural) {

    BrainStormNet *nn = (BrainStormNet *)neural;
    
    int tensor_length = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        int dim = 1;
        for (int i=0; i<nn->dense->costWeightDerivatives->rank; i++) {
            dim = dim * nn->dense->costWeightDerivatives->shape[l][i][0];
        }
        tensor_length = tensor_length + dim;
    }
    memset(nn->dense->costWeightDerivatives->val, 0.0f, tensor_length*sizeof(float));
    memset(nn->dense->batchCostWeightDeriv->val, 0.0f, tensor_length*sizeof(float));
    
    tensor_length = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        int dim = 1;
        for (int i=0; i<nn->dense->costBiasDerivatives->rank; i++) {
            dim = dim * nn->dense->costBiasDerivatives->shape[l][i][0];
        }
        tensor_length = tensor_length + dim;
    }
    memset(nn->dense->costBiasDerivatives->val, 0.0f, tensor_length*sizeof(float));
    memset(nn->dense->batchCostBiasDeriv->val, 0.0f, tensor_length*sizeof(float));
}

void grad_descent_update_in_dense_net(void * _Nullable neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update weights
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->weights->val[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) -
                (nn->dense->train->gradient_descent->learning_rate/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)];
            }
        }
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense->biases->shape[l][0][0];
        
        for (int i=0; i<n; i++) {
            nn->dense->biases->val[stride+i] = nn->dense->biases->val[stride+i] - (nn->dense->train->gradient_descent->learning_rate/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i];
        }
        stride = stride + n;
    }
    
}

void momentum_update_in_dense_net(void * _Nullable neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update weights
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        float coeff[m][n];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = (nn->dense->train->momentum->learning_rate/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)];
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->weightsVelocity->val[stride+((i*n)+j)] = nn->dense->train->momentum->momentum_coefficient *
                nn->dense->weightsVelocity->val[stride+((i*n)+j)] - coeff[i][j];
                nn->dense->weights->val[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                nn->dense->weightsVelocity->val[stride+((i*n)+j)];
            }
        }
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense->biases->shape[l][0][0];
        
        float coeff[n];
        for (int i=0; i<n; i++) {
            coeff[i] = (nn->dense->train->momentum->learning_rate/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i];
        }
        
        for (int i=0; i<n; i++) {
            nn->dense->biasesVelocity->val[stride+i] = nn->dense->train->momentum->momentum_coefficient *
            nn->dense->biasesVelocity->val[stride+i] - coeff[i];
            nn->dense->biases->val[stride+i] = nn->dense->biases->val[stride+i] + nn->dense->biasesVelocity->val[stride+i];
        }
        stride = stride + n;
    }
}

void ada_grad_update_in_dense_net(void * _Nullable neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update weights
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        float coeff[m][n];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)] = nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)] +
                ( ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)]) * ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)]) );
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -( nn->dense->train->ada_grad->learning_rate/(nn->dense->train->ada_grad->delta+sqrtf(nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)])) ) *
                ( (1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)] );
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->weights->val[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                coeff[i][j];
            }
        }
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense->biases->shape[l][0][0];
        
        float coeff[n];
        
        for (int i=0; i<n; i++) {
            nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated->val[stride+i] = nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated->val[stride+i] +
            ( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i]) * ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i]) );
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -( nn->dense->train->ada_grad->learning_rate/(nn->dense->train->ada_grad->delta+sqrtf(nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated->val[stride+i])) ) *
            ( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i]) );
        }
        
        for (int i=0; i<n; i++) {
            nn->dense->biases->val[stride+i] = nn->dense->biases->val[stride+i] + coeff[i];
        }
        stride = stride + n;
    }
}

void rms_prop_update_in_dense_net(void * _Nonnull neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update weights
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        float coeff[m][n];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)] = (nn->dense->train->rms_prop->decayRate*nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)]) +
                (1.0f-nn->dense->train->rms_prop->decayRate)*( ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)]) * ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)]) );
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -( nn->dense->train->rms_prop->learning_rate/(sqrtf(nn->dense->train->rms_prop->delta+nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)])) ) *
                ( (1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)] );
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->weights->val[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                coeff[i][j];
            }
        }
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense->biases->shape[l][0][0];
        
        // Adpative learning rate
        float coeff[n];
        
        for (int i=0; i<n; i++) {
            nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated->val[stride+i] = (nn->dense->train->rms_prop->decayRate*nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated->val[stride+i]) +
            (1.0-nn->dense->train->rms_prop->decayRate)*( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i]) * ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i]) );
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -( nn->dense->train->rms_prop->learning_rate/(sqrtf(nn->dense->train->rms_prop->delta+nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated->val[stride+i])) ) *
            ( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i]) );
        }
        
        for (int i=0; i<n; i++) {
            nn->dense->biases->val[stride+i] = nn->dense->biases->val[stride+i] + coeff[i];
        }
        stride = stride + n;
    }
}

void adam_update_in_dense_net(void * _Nonnull neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update weights
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        float coeff[m][n];
        
        nn->dense->train->adam->time++;
        float s_hat[m][n];
        float r_hat[m][n];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                // Update biased first moment estimate
                nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate->val[stride+((i*n)+j)] = (nn->dense->train->adam->decayRate1*nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate->val[stride+((i*n)+j)]) +
                (1.0f-nn->dense->train->adam->decayRate1)*( ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)]) );
                // Update biased second moment estimate
                nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate->val[stride+((i*n)+j)] = (nn->dense->train->adam->decayRate2*nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate->val[stride+((i*n)+j)]) +
                (1.0f-nn->dense->train->adam->decayRate2)*( ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)]) * ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[stride+((i*n)+j)]) );
                
                // Correct bias in first moment
                s_hat[i][j] = nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate->val[stride+((i*n)+j)] / (1.0f - powf(nn->dense->train->adam->decayRate1, (float)nn->dense->train->adam->time));
                // Correct bias in second moment
                r_hat[i][j] = nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate->val[stride+((i*n)+j)] / (1.0f - powf(nn->dense->train->adam->decayRate2, (float)nn->dense->train->adam->time));
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -nn->dense->train->adam->stepSize*( s_hat[i][j] / (sqrtf(r_hat[i][j]+nn->dense->train->adam->delta)) );
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->weights->val[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                coeff[i][j];
            }
        }
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense->biases->shape[l][0][0];
        
        float coeff[n];
        
        float s_hat[n];
        float r_hat[n];
        for (int i=0; i<n; i++) {
            // Update biased first moment estimate
            nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate->val[stride+i] = (nn->dense->train->adam->decayRate1*nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate->val[stride+i]) +
            (1.0-nn->dense->train->adam->decayRate1)*( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i]) );
            // Update biased second moment estimate
            nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate->val[stride+i] = (nn->dense->train->adam->decayRate2*nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate->val[stride+i]) +
            (1.0-nn->dense->train->adam->decayRate2)*( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i]) * ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[stride+i]) );
            
            // Correct bias in first moment
            s_hat[i] = nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate->val[stride+i] / (1.0f - powf(nn->dense->train->adam->decayRate1, (float)nn->dense->train->adam->time));
            // Correct bias in second moment
            r_hat[i] = nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate->val[stride+i] / (1.0f - powf(nn->dense->train->adam->decayRate2, (float)nn->dense->train->adam->time));
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -nn->dense->train->adam->stepSize*( s_hat[i] / (sqrtf(r_hat[i]+nn->dense->train->adam->delta)) );
        }
        
        for (int i=0; i<n; i++) {
            nn->dense->biases->val[stride+i] = nn->dense->biases->val[stride+i] + coeff[i];
        }
        stride = stride + n;
    }
}


void gradientDescentOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    static bool  firstTime = true;
    static ptr_inference_func inference = NULL;
    static ptr_backpropag_func backpropagation = NULL;
    static ptr_batch_accumul_func batch_accumulation = NULL;
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (firstTime) {
        if (nn->is_dense_network) {
            ptr_init_func = init_in_dense_net;
            ptr_gradient_descent_update_func = grad_descent_update_in_dense_net;
            inference = inference_in_dense_net;
            backpropagation = backpropag_in_dense_net;
            batch_accumulation = batch_accumulation_in_dense_net;
        } else if (nn->is_conv2d_network) {
            //TODO:
        }
        firstTime = false;
    }
    
    nn->batch = mini_batch;
    
    ptr_init_func((void *)nn);
    miniBatchLoop((void *)nn, batch_size, inference, backpropagation, batch_accumulation);
    ptr_gradient_descent_update_func((void *)nn, batch_size);
}

void momentumOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    static bool firstTime = true;
    static ptr_inference_func inference = NULL;
    static ptr_backpropag_func backpropagation = NULL;
    static ptr_batch_accumul_func batch_accumulation = NULL;
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (firstTime) {
        if (nn->is_dense_network) {
            ptr_init_func = init_in_dense_net;
            ptr_momentum_update_func = momentum_update_in_dense_net;
            inference = inference_in_dense_net;
            backpropagation = backpropag_in_dense_net;
            batch_accumulation = batch_accumulation_in_dense_net;
        } else if (nn->is_conv2d_network) {
            //TODO:
        }
        firstTime = false;
    }
    
    nn->batch = mini_batch;
    
    ptr_init_func((void *)nn);
    miniBatchLoop((void *)nn, batch_size, inference, backpropagation, batch_accumulation);
    ptr_momentum_update_func((void *)nn, batch_size);
}

void adaGradOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
 
    static bool firstTime = true;
    static ptr_inference_func inference = NULL;
    static ptr_backpropag_func backpropagation = NULL;
    static ptr_batch_accumul_func batch_accumulation = NULL;
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (firstTime) {
        if (nn->is_dense_network) {
            ptr_init_func = init_in_dense_net;
            ptr_ada_grad_update_func = ada_grad_update_in_dense_net;
            inference = inference_in_dense_net;
            backpropagation = backpropag_in_dense_net;
            batch_accumulation = batch_accumulation_in_dense_net;
        } else if (nn->is_conv2d_network) {
            //TODO:
        }
        firstTime = false;
    }
    
    nn->batch = mini_batch;
    
    ptr_init_func((void *)nn);
    miniBatchLoop((void *)nn, batch_size, inference, backpropagation, batch_accumulation);
    ptr_ada_grad_update_func((void *)neural, batch_size);
}

void rmsPropOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    static bool firstTime = true;
    static ptr_inference_func inference = NULL;
    static ptr_backpropag_func backpropagation = NULL;
    static ptr_batch_accumul_func batch_accumulation = NULL;
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (firstTime) {
        if (nn->is_dense_network) {
            ptr_init_func = init_in_dense_net;
            ptr_rms_prop_update_func = rms_prop_update_in_dense_net;
            inference = inference_in_dense_net;
            backpropagation = backpropag_in_dense_net;
            batch_accumulation = batch_accumulation_in_dense_net;
        } else if (nn->is_conv2d_network) {
            //TODO:
        }
        firstTime = false;
    }
    
    nn->batch = mini_batch;
    
    ptr_init_func((void *)nn);
    miniBatchLoop((void *)nn, batch_size, inference, backpropagation, batch_accumulation);
    ptr_rms_prop_update_func((void *)nn, batch_size);
}

void adamOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    static bool firstTime = true;
    static ptr_inference_func inference = NULL;
    static ptr_backpropag_func backpropagation = NULL;
    static ptr_batch_accumul_func batch_accumulation = NULL;
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (firstTime) {
        if (nn->is_dense_network) {
            ptr_init_func = init_in_dense_net;
            ptr_adam_update_func = adam_update_in_dense_net;
            inference = inference_in_dense_net;
            backpropagation = backpropag_in_dense_net;
            batch_accumulation = batch_accumulation_in_dense_net;
        } else if (nn->is_conv2d_network) {
            //TODO:
        }
        firstTime = false;
    }
    
    nn->batch = mini_batch;
    
    ptr_init_func((void *)nn);
    miniBatchLoop((void *)nn, batch_size, inference, backpropagation, batch_accumulation);
    ptr_adam_update_func((void *)nn, batch_size);
}
