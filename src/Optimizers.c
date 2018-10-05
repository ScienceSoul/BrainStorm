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
#include "Conv2DNetOps.h"

typedef enum optimizer {
    GRADIENT_DESCENT=1,
    MOMENTUM,
    ADAGRAD,
    RMSPROP,
    ADAM
} optimizer;

static ptr_inference_func inference = NULL;
static ptr_backpropag_func backpropagation = NULL;
static ptr_batch_accumul_func batch_accumulation = NULL;

static void (* _Nullable ptr_init_func)(void * _Nonnull neural) = NULL;
static void (* _Nullable ptr_gradient_descent_update_func)(void * _Nonnull  neural, unsigned int batch_size) = NULL;
static void (* _Nullable ptr_momentum_update_func)(void * _Nonnull  neural, unsigned int batch_size) = NULL;
static void (* _Nullable ptr_ada_grad_update_func)(void * _Nonnull  neural, unsigned int batch_size) = NULL;
static void (* _Nullable ptr_rms_prop_update_func)(void * _Nonnull  neural, unsigned int batch_size) = NULL;
static void (* _Nullable ptr_adam_update_func)(void * _Nonnull neural, unsigned int batch_size) = NULL;

static void init_in_dense_net(void * _Nonnull neural) {

    static int firstTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    static unsigned int w_length = 0;
    static unsigned int b_length = 0;
    if (firstTime) {
        
        for (int l=0; l<nn->network_num_layers-1; l++) {
            int dim = 1;
            for (int i=0; i<nn->dense->costWeightDerivatives->rank; i++) {
                dim = dim * nn->dense->costWeightDerivatives->shape[l][i][0];
            }
            w_length = w_length + dim;
        }
        
        for (int l=0; l<nn->network_num_layers-1; l++) {
            int dim = 1;
            for (int i=0; i<nn->dense->costBiasDerivatives->rank; i++) {
                dim = dim * nn->dense->costBiasDerivatives->shape[l][i][0];
            }
            b_length = b_length + dim;
        }
        
        firstTime = false;
    }
    
    memset(nn->dense->costWeightDerivatives->val, 0.0f, w_length*sizeof(float));
    memset(nn->dense->batchCostWeightDeriv->val, 0.0f, w_length*sizeof(float));
    
    memset(nn->dense->costBiasDerivatives->val, 0.0f, b_length*sizeof(float));
    memset(nn->dense->batchCostBiasDeriv->val, 0.0f, b_length*sizeof(float));
}

static void init_in_conv2d_net(void * _Nonnull neural) {
    
    static bool firstTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    static unsigned int w_conv_length = 0;
    static unsigned int b_conv_length = 0;
    static unsigned int w_dense_length = 0;
    static unsigned int b_dense_length = 0;
    
    if (firstTime) {
        
        for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
            int dim = 1;
            for (int i=0; i<nn->conv2d->conv_costWeightDerivatives->rank; i++) {
                dim = dim * nn->conv2d->conv_costWeightDerivatives->shape[l][i][0];
            }
            w_conv_length = w_conv_length + dim;
            
            dim = 1;
            for (int i=0; i<nn->conv2d->conv_costBiasDerivatives->rank; i++) {
                dim = dim * nn->conv2d->conv_costBiasDerivatives->shape[l][i][0];
            }
            b_conv_length = b_conv_length + dim;
        }
        
        for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
            int dim = 1;
            for (int i=0; i<nn->conv2d->dense_costWeightDerivatives->rank; i++) {
                dim = dim * nn->conv2d->dense_costWeightDerivatives->shape[l][i][0];
            }
            w_dense_length = w_dense_length + dim;
            
            dim = 1;
            for (int i=0; i<nn->conv2d->dense_costBiasDerivatives->rank; i++) {
                dim = dim * nn->conv2d->dense_costBiasDerivatives->shape[l][i][0];
            }
            b_dense_length = b_dense_length + dim;
        }
        
        firstTime = false;
    }
    
    memset(nn->conv2d->conv_costWeightDerivatives->val, 0.0f, w_conv_length*sizeof(float));
    memset(nn->conv2d->conv_batchCostWeightDeriv->val, 0.0f, w_conv_length*sizeof(float));
    memset(nn->conv2d->conv_costBiasDerivatives->val, 0.0f, b_conv_length*sizeof(float));
    memset(nn->conv2d->conv_batchCostBiasDeriv->val, 0.0f, b_conv_length*sizeof(float));
    
    memset(nn->conv2d->dense_costWeightDerivatives->val, 0.0f, w_dense_length*sizeof(float));
    memset(nn->conv2d->dense_batchCostWeightDeriv->val, 0.0f, w_dense_length*sizeof(float));
    memset(nn->conv2d->dense_costBiasDerivatives->val, 0.0f, b_dense_length*sizeof(float));
    memset(nn->conv2d->dense_batchCostBiasDeriv->val, 0.0f, b_dense_length*sizeof(float));
}

static void grad_descent_update_in_dense_net(void * _Nullable neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update weights
    unsigned int offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->weights->val[offset+((i*n)+j)] = nn->regularizer[l](neural, nn->dense->weights->val, nn->dense->parameters->eta, nn->dense->parameters->lambda, i, j, n, offset, 0, 0) -
                (nn->dense->train->gradient_descent->learning_rate/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)];
            }
        }
        offset = offset + (m * n);
    }
    
    // Update biases
    offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense->biases->shape[l][0][0];
        
        for (int i=0; i<n; i++) {
            nn->dense->biases->val[offset+i] = nn->dense->biases->val[offset+i] - (nn->dense->train->gradient_descent->learning_rate/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i];
        }
        offset = offset + n;
    }
}

static void grad_descent_update_in_conv2d_net(void * _Nonnull neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update the weights and biases at the convolution layers
    int offset_w = 0;
    int offset_b = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned int p = nn->conv2d->conv_weights->shape[l][0][0];
        unsigned int q = nn->conv2d->conv_weights->shape[l][1][0];
        unsigned int kh = nn->conv2d->conv_weights->shape[l][2][0];
        unsigned int kw = nn->conv2d->conv_weights->shape[l][3][0];
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        nn->conv2d->conv_weights->val[offset_w+(stride1+(stride2+(u*kw+v)))] =
                           nn->regularizer[l](neural, nn->conv2d->conv_weights->val, nn->conv2d->parameters->eta, nn->conv2d->parameters->lambda, u, v, kw, offset_w, stride1, stride2) -
                        (nn->conv2d->train->gradient_descent->learning_rate/(float)batch_size)*nn->conv2d->conv_costWeightDerivatives->val[offset_w+(stride1+(stride2+(u*kw+v)))];
                    }
                }
                stride2 = stride2 + (kh * kw);
            }
            stride1 = stride1 + (q * kh * kw);
        }
        
         for (int ll=0; ll<q; ll++) {
             nn->conv2d->conv_biases->val[offset_b+ll] = nn->conv2d->conv_biases->val[offset_b+ll] - (nn->conv2d->train->gradient_descent->learning_rate/(float)batch_size)*nn->conv2d->conv_costBiasDerivatives->val[offset_b+ll];
         }
        
        offset_w = offset_w + (p * q * kh * kw);
        offset_b = offset_b + q;
    }
    
    // Update the weights and biases at the fully connected layers
    offset_w = 0;
    int idx = nn->conv2d->num_conv2d_layers;
    for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
        unsigned int m = nn->conv2d->dense_weights->shape[l][0][0];
        unsigned int n = nn->conv2d->dense_weights->shape[l][1][0];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->conv2d->dense_weights->val[offset_w+((i*n)+j)] = nn->regularizer[idx](neural, nn->conv2d->dense_weights->val, nn->conv2d->parameters->eta, nn->conv2d->parameters->lambda, i, j, n, offset_w, 0, 0) - (nn->conv2d->train->gradient_descent->learning_rate/(float)batch_size)*nn->conv2d->dense_costWeightDerivatives->val[offset_w+((i*n)+j)];
            }
        }
        offset_w = offset_w + (m * n);
        idx++;
    }
    
    offset_b = 0;
    for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
        unsigned int n = nn->conv2d->dense_biases->shape[l][0][0];
        
        for (int i=0; i<n; i++) {
            nn->conv2d->dense_biases->val[offset_b+i] = nn->conv2d->dense_biases->val[offset_b+i] - (nn->conv2d->train->gradient_descent->learning_rate/(float)batch_size)*nn->conv2d->dense_costBiasDerivatives->val[offset_b+i];
        }
        offset_b = offset_b + n;
    }
    
    // -----------------------------------------------------------
    // Update also the convolution matrices with the new weights
    // -----------------------------------------------------------
    nn->flip_kernels((void *)nn);
    nn->conv_mat_update((void *)nn);
}

static void momentum_update_in_dense_net(void * _Nullable neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update weights
    unsigned int offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        float coeff[m][n];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = (nn->dense->train->momentum->learning_rate/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)];
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->weightsVelocity->val[offset+((i*n)+j)] = nn->dense->train->momentum->momentum_coefficient * nn->dense->weightsVelocity->val[offset+((i*n)+j)] - coeff[i][j];
                nn->dense->weights->val[offset+((i*n)+j)] = nn->regularizer[l](neural, nn->dense->weights->val,nn->dense->parameters->eta, nn->dense->parameters->lambda, i, j, n, offset, 0, 0) +
                nn->dense->weightsVelocity->val[offset+((i*n)+j)];
            }
        }
        offset = offset + (m * n);
    }
    
    // Update biases
    offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense->biases->shape[l][0][0];
        
        float coeff[n];
        for (int i=0; i<n; i++) {
            coeff[i] = (nn->dense->train->momentum->learning_rate/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i];
        }
        
        for (int i=0; i<n; i++) {
            nn->dense->biasesVelocity->val[offset+i] = nn->dense->train->momentum->momentum_coefficient *
            nn->dense->biasesVelocity->val[offset+i] - coeff[i];
            nn->dense->biases->val[offset+i] = nn->dense->biases->val[offset+i] + nn->dense->biasesVelocity->val[offset+i];
        }
        offset = offset + n;
    }
}

static void momentum_update_in_conv2d_net(void * _Nonnull neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update the weights and biases at the convolution layers
    int offset_w = 0;
    int offset_b = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned int p = nn->conv2d->conv_weights->shape[l][0][0];
        unsigned int q = nn->conv2d->conv_weights->shape[l][1][0];
        unsigned int kh = nn->conv2d->conv_weights->shape[l][2][0];
        unsigned int kw = nn->conv2d->conv_weights->shape[l][3][0];
        float coeff_w[kh][kw];
        float coeff_b[q];
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        coeff_w[u][v] =  (nn->conv2d->train->momentum->learning_rate/(float)batch_size)*nn->conv2d->conv_costWeightDerivatives->val[offset_w+(stride1+(stride2+(u*kw+v)))];
                    }
                }
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        nn->conv2d->conv_weightsVelocity->val[offset_w+(stride1+(stride2+(u*kw+v)))] =
                            nn->conv2d->train->momentum->momentum_coefficient * nn->conv2d->conv_weightsVelocity->val[offset_w+(stride1+(stride2+(u*kw+v)))] - coeff_w[u][v];
                        nn->conv2d->conv_weights->val[offset_w+(stride1+(stride2+(u*kw+v)))] = nn->regularizer[l](neural, nn->conv2d->conv_weights->val, nn->conv2d->parameters->eta, nn->conv2d->parameters->lambda, u, v, kw, offset_w, stride1, stride2) + nn->conv2d->conv_weightsVelocity->val[offset_w+(stride1+(stride2+(u*kw+v)))];
                    }
                }
                stride2 = stride2 + (kh * kw);
            }
            stride1 = stride1 + (q * kh * kw);
        }
        
        for (int ll=0; ll<q; ll++) {
            coeff_b[ll] = (nn->conv2d->train->momentum->learning_rate/(float)batch_size)*nn->conv2d->conv_costBiasDerivatives->val[offset_b+ll];
        }
        for (int ll=0; ll<q; ll++) {
            nn->conv2d->conv_biasesVelocity->val[offset_b+ll] = nn->conv2d->train->momentum->momentum_coefficient * nn->conv2d->conv_biasesVelocity->val[offset_b+ll] - coeff_b[ll];
            nn->conv2d->conv_biases->val[offset_b+ll] = nn->conv2d->conv_biases->val[offset_b+ll] + nn->conv2d->conv_biasesVelocity->val[offset_b+ll];
        }
        
        offset_w = offset_w + (p * q * kh * kw);
        offset_b = offset_b + q;
    }
    
    // Update the weights and biases at the fully connected layers
    offset_w = 0;
    int idx = nn->conv2d->num_conv2d_layers;
    for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
        unsigned int m = nn->conv2d->dense_weights->shape[l][0][0];
        unsigned int n = nn->conv2d->dense_weights->shape[l][1][0];
        
        float coeff[m][n];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = (nn->conv2d->train->momentum->learning_rate/(float)batch_size)*nn->conv2d->dense_costWeightDerivatives->val[offset_w+((i*n)+j)];
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->conv2d->dense_weightsVelocity->val[offset_w+((i*n)+j)] = nn->conv2d->train->momentum->momentum_coefficient * nn->conv2d->dense_weightsVelocity->val[offset_w+((i*n)+j)] - coeff[i][j];
                nn->conv2d->dense_weights->val[offset_w+((i*n)+j)] = nn->regularizer[idx](neural, nn->conv2d->dense_weights->val, nn->conv2d->parameters->eta, nn->conv2d->parameters->lambda, i, j, n, offset_w, 0, 0) + nn->conv2d->dense_weightsVelocity->val[offset_w+((i*n)+j)];
            }
        }
        offset_w = offset_w + (m * n);
        idx++;
    }
    
    offset_b = 0;
    for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
        unsigned int n = nn->conv2d->dense_biases->shape[l][0][0];
        
        float coeff[n];
        for (int i=0; i<n; i++) {
            coeff[i] = (nn->conv2d->train->momentum->learning_rate/(float)batch_size)*nn->conv2d->dense_costBiasDerivatives->val[offset_b+i];
        }
        
        for (int i=0; i<n; i++) {
            nn->conv2d->dense_biasesVelocity->val[offset_b+i] = nn->conv2d->train->momentum->momentum_coefficient * nn->conv2d->dense_biasesVelocity->val[offset_b+i] - coeff[i];
            nn->conv2d->dense_biases->val[offset_b+i] = nn->conv2d->dense_biases->val[offset_b+i] + nn->conv2d->dense_biasesVelocity->val[offset_b+i];
        }
        offset_b = offset_b + n;
    }
    
    // -----------------------------------------------------------
    // Update also the convolution matrices with the new weights
    // -----------------------------------------------------------
    nn->flip_kernels((void *)nn);
    nn->conv_mat_update((void *)nn);
}

static void ada_grad_update_in_dense_net(void * _Nullable neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update weights
    unsigned int offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        float coeff[m][n];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated->val[offset+((i*n)+j)] = nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated->val[offset+((i*n)+j)] +
                ( ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)]) * ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)]) );
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -( nn->dense->train->ada_grad->learning_rate/(nn->dense->train->ada_grad->delta+sqrtf(nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated->val[offset+((i*n)+j)])) ) *
                ( (1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)] );
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->weights->val[offset+((i*n)+j)] = nn->regularizer[l](neural, nn->dense->weights->val, nn->dense->parameters->eta, nn->dense->parameters->lambda, i, j, n, offset, 0, 0) +
                coeff[i][j];
            }
        }
        offset = offset + (m * n);
    }
    
    // Update biases
    offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense->biases->shape[l][0][0];
        
        float coeff[n];
        
        for (int i=0; i<n; i++) {
            nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated->val[offset+i] = nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated->val[offset+i] +
            ( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i]) * ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i]) );
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -( nn->dense->train->ada_grad->learning_rate/(nn->dense->train->ada_grad->delta+sqrtf(nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated->val[offset+i])) ) *
            ( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i]) );
        }
        
        for (int i=0; i<n; i++) {
            nn->dense->biases->val[offset+i] = nn->dense->biases->val[offset+i] + coeff[i];
        }
        offset = offset + n;
    }
}

static void rms_prop_update_in_dense_net(void * _Nonnull neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update weights
    unsigned int offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        float coeff[m][n];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated->val[offset+((i*n)+j)] = (nn->dense->train->rms_prop->decayRate*nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated->val[offset+((i*n)+j)]) +
                (1.0f-nn->dense->train->rms_prop->decayRate)*( ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)]) * ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)]) );
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -( nn->dense->train->rms_prop->learning_rate/(sqrtf(nn->dense->train->rms_prop->delta+nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated->val[offset+((i*n)+j)])) ) *
                ( (1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)] );
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->weights->val[offset+((i*n)+j)] = nn->regularizer[l](neural, nn->dense->weights->val, nn->dense->parameters->eta, nn->dense->parameters->lambda, i, j, n, offset, 0, 0) +
                coeff[i][j];
            }
        }
        offset = offset + (m * n);
    }
    
    // Update biases
    offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense->biases->shape[l][0][0];
        
        // Adpative learning rate
        float coeff[n];
        
        for (int i=0; i<n; i++) {
            nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated->val[offset+i] = (nn->dense->train->rms_prop->decayRate*nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated->val[offset+i]) +
            (1.0-nn->dense->train->rms_prop->decayRate)*( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i]) * ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i]) );
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -( nn->dense->train->rms_prop->learning_rate/(sqrtf(nn->dense->train->rms_prop->delta+nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated->val[offset+i])) ) *
            ( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i]) );
        }
        
        for (int i=0; i<n; i++) {
            nn->dense->biases->val[offset+i] = nn->dense->biases->val[offset+i] + coeff[i];
        }
        offset = offset + n;
    }
}

static void adam_update_in_dense_net(void * _Nonnull neural, unsigned int batch_size) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Update weights
    unsigned int offset = 0;
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
                nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate->val[offset+((i*n)+j)] = (nn->dense->train->adam->decayRate1*nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate->val[offset+((i*n)+j)]) +
                (1.0f-nn->dense->train->adam->decayRate1)*( ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)]) );
                // Update biased second moment estimate
                nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate->val[offset+((i*n)+j)] = (nn->dense->train->adam->decayRate2*nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate->val[offset+((i*n)+j)]) +
                (1.0f-nn->dense->train->adam->decayRate2)*( ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)]) * ((1.0f/(float)batch_size)*nn->dense->costWeightDerivatives->val[offset+((i*n)+j)]) );
                
                // Correct bias in first moment
                s_hat[i][j] = nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate->val[offset+((i*n)+j)] / (1.0f - powf(nn->dense->train->adam->decayRate1, (float)nn->dense->train->adam->time));
                // Correct bias in second moment
                r_hat[i][j] = nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate->val[offset+((i*n)+j)] / (1.0f - powf(nn->dense->train->adam->decayRate2, (float)nn->dense->train->adam->time));
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -nn->dense->train->adam->stepSize*( s_hat[i][j] / (sqrtf(r_hat[i][j]+nn->dense->train->adam->delta)) );
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->weights->val[offset+((i*n)+j)] = nn->regularizer[l](neural, nn->dense->weights->val, nn->dense->parameters->eta, nn->dense->parameters->lambda, i, j, n, offset, 0, 0) +
                coeff[i][j];
            }
        }
        offset = offset + (m * n);
    }
    
    // Update biases
    offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense->biases->shape[l][0][0];
        
        float coeff[n];
        
        float s_hat[n];
        float r_hat[n];
        for (int i=0; i<n; i++) {
            // Update biased first moment estimate
            nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate->val[offset+i] = (nn->dense->train->adam->decayRate1*nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate->val[offset+i]) +
            (1.0-nn->dense->train->adam->decayRate1)*( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i]) );
            // Update biased second moment estimate
            nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate->val[offset+i] = (nn->dense->train->adam->decayRate2*nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate->val[offset+i]) +
            (1.0-nn->dense->train->adam->decayRate2)*( ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i]) * ((1.0f/(float)batch_size)*nn->dense->costBiasDerivatives->val[offset+i]) );
            
            // Correct bias in first moment
            s_hat[i] = nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate->val[offset+i] / (1.0f - powf(nn->dense->train->adam->decayRate1, (float)nn->dense->train->adam->time));
            // Correct bias in second moment
            r_hat[i] = nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate->val[offset+i] / (1.0f - powf(nn->dense->train->adam->decayRate2, (float)nn->dense->train->adam->time));
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -nn->dense->train->adam->stepSize*( s_hat[i] / (sqrtf(r_hat[i]+nn->dense->train->adam->delta)) );
        }
        
        for (int i=0; i<n; i++) {
            nn->dense->biases->val[offset+i] = nn->dense->biases->val[offset+i] + coeff[i];
        }
        offset = offset + n;
    }
}

static void set_func_ptr(void * _Nonnull neural, optimizer optimizer) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (nn->is_dense_network) {
        ptr_init_func = init_in_dense_net;
        
        if (optimizer == GRADIENT_DESCENT) {
            ptr_gradient_descent_update_func = grad_descent_update_in_dense_net;
        } else if (optimizer == MOMENTUM) {
            ptr_momentum_update_func = momentum_update_in_dense_net;
        } else if (optimizer == ADAGRAD) {
            ptr_ada_grad_update_func = ada_grad_update_in_dense_net;
        } else if (optimizer == RMSPROP) {
            ptr_rms_prop_update_func = rms_prop_update_in_dense_net;
        } else if (optimizer == ADAM) {
            ptr_adam_update_func = adam_update_in_dense_net;
        }
        
        inference = inference_in_dense_net;
        backpropagation = backpropag_in_dense_net;
        batch_accumulation = batch_accumulation_in_dense_net;
    } else if (nn->is_conv2d_network) {
        ptr_init_func = init_in_conv2d_net;
        
        if (optimizer == GRADIENT_DESCENT) {
            ptr_gradient_descent_update_func = grad_descent_update_in_conv2d_net;
        } else if (optimizer == MOMENTUM) {
            ptr_momentum_update_func = momentum_update_in_conv2d_net;
        } else if (optimizer == ADAGRAD) {
            
        } else if (optimizer == RMSPROP) {
            
        } else if (optimizer == ADAM) {
            
        }
        
        inference = inference_in_conv2d_net;
        backpropagation = backpropag_in_conv2d_net;
        batch_accumulation = batch_accumulation_in_conv2d_net;
    }
}

void gradientDescentOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    static bool  firstTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    optimizer optimizer = GRADIENT_DESCENT;
    if (firstTime) {
        set_func_ptr(neural, optimizer);
        firstTime = false;
    }
    
    nn->batch = mini_batch;
    ptr_init_func((void *)nn);
    miniBatchLoop((void *)nn, batch_size, inference, backpropagation, batch_accumulation);
    ptr_gradient_descent_update_func((void *)nn, batch_size);
}

void momentumOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    static bool firstTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    optimizer optimizer = MOMENTUM;
    if (firstTime) {
        set_func_ptr(neural, optimizer);
        firstTime = false;
    }
    
    nn->batch = mini_batch;
    ptr_init_func((void *)nn);
    miniBatchLoop((void *)nn, batch_size, inference, backpropagation, batch_accumulation);
    ptr_momentum_update_func((void *)nn, batch_size);
}

void adaGradOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
 
    static bool firstTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    optimizer optimizer = ADAGRAD;
    if (firstTime) {
        set_func_ptr(neural, optimizer);
        firstTime = false;
    }
    
    nn->batch = mini_batch;
    ptr_init_func((void *)nn);
    miniBatchLoop((void *)nn, batch_size, inference, backpropagation, batch_accumulation);
    ptr_ada_grad_update_func((void *)neural, batch_size);
}

void rmsPropOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    static bool firstTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    optimizer optimizer = RMSPROP;
    if (firstTime) {
        set_func_ptr(neural, optimizer);
        firstTime = false;
    }
    
    nn->batch = mini_batch;
    ptr_init_func((void *)nn);
    miniBatchLoop((void *)nn, batch_size, inference, backpropagation, batch_accumulation);
    ptr_rms_prop_update_func((void *)nn, batch_size);
}

void adamOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    static bool firstTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    optimizer optimizer = ADAM;
    if (firstTime) {
        set_func_ptr(neural, optimizer);
        firstTime = false;
    }
    
    nn->batch = mini_batch;
    ptr_init_func((void *)nn);
    miniBatchLoop((void *)nn, batch_size, inference, backpropagation, batch_accumulation);
    ptr_adam_update_func((void *)nn, batch_size);
}
