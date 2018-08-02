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
#include "NetworkOps.h"

void init(void *neural) {

    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    int tensor_length = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        int dim = 1;
        for (int i=0; i<nn->dense_costWeightDerivatives->rank; i++) {
            dim = dim * nn->dense_costWeightDerivatives->shape[l][i][0];
        }
        tensor_length = tensor_length + dim;
    }
    memset(nn->dense_costWeightDerivatives->val, 0.0f, tensor_length*sizeof(float));
    memset(nn->dense_batchCostWeightDeriv->val, 0.0f, tensor_length*sizeof(float));
    
    tensor_length = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        int dim = 1;
        for (int i=0; i<nn->dense_costBiasDerivatives->rank; i++) {
            dim = dim * nn->dense_costBiasDerivatives->shape[l][i][0];
        }
        tensor_length = tensor_length + dim;
    }
    memset(nn->dense_costBiasDerivatives->val, 0.0f, tensor_length*sizeof(float));
    memset(nn->dense_batchCostBiasDeriv->val, 0.0f, tensor_length*sizeof(float));
}

void gradientDescentOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    nn->batch = mini_batch;
    
    init((void *)nn);
    
    miniBatchLoop((void *)nn, batch_size);
    
    // Update weights
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense_weights->shape[l][0][0];
        unsigned int n = nn->dense_weights->shape[l][1][0];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense_weights->val[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) -
                     (nn->train->gradient_descent->learning_rate/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)];
            }
        }
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense_biases->shape[l][0][0];
        
        for (int i=0; i<n; i++) {
            nn->dense_biases->val[stride+i] = nn->dense_biases->val[stride+i] - (nn->train->gradient_descent->learning_rate/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i];
        }
        stride = stride + n;
    }
}

void momentumOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    nn->batch = mini_batch;
    
    init((void *)nn);
    
    miniBatchLoop((void *)nn, batch_size);
    
    // Update weights
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense_weights->shape[l][0][0];
        unsigned int n = nn->dense_weights->shape[l][1][0];
        
        float coeff[m][n];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = (nn->train->momentum->learning_rate/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)];
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense_weightsVelocity->val[stride+((i*n)+j)] = nn->train->momentum->momentum_coefficient *
                                                 nn->dense_weightsVelocity->val[stride+((i*n)+j)] - coeff[i][j];
                nn->dense_weights->val[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                                                nn->dense_weightsVelocity->val[stride+((i*n)+j)];
            }
        }
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense_biases->shape[l][0][0];
        
        // Adpative learning rate
        float coeff[n];
        
        for (int i=0; i<n; i++) {
            coeff[i] = (nn->train->momentum->learning_rate/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i];
        }
        
        for (int i=0; i<n; i++) {
            nn->dense_biasesVelocity->val[stride+i] = nn->train->momentum->momentum_coefficient *
                                                nn->dense_biasesVelocity->val[stride+i] - coeff[i];
            nn->dense_biases->val[stride+i] = nn->dense_biases->val[stride+i] + nn->dense_biasesVelocity->val[stride+i];
        }
        stride = stride + n;
    }
}

void adaGradOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
 
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    nn->batch = mini_batch;
    
    init((void *)nn);
    
    miniBatchLoop((void *)nn, batch_size);
    
    // Update weights
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense_weights->shape[l][0][0];
        unsigned int n = nn->dense_weights->shape[l][1][0];
        
        float coeff[m][n];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->train->ada_grad->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)] = nn->train->ada_grad->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)] + ( ((1.0f/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)]) * ((1.0f/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)]) );
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -( nn->train->ada_grad->learning_rate/(nn->train->ada_grad->delta+sqrtf(nn->train->ada_grad->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)])) ) * ( (1.0f/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)] );
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense_weights->val[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                                                coeff[i][j];
            }
        }
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense_biases->shape[l][0][0];

        float coeff[n];
        
        for (int i=0; i<n; i++) {
            nn->train->ada_grad->costBiasDerivativeSquaredAccumulated->val[stride+i] = nn->train->ada_grad->costBiasDerivativeSquaredAccumulated->val[stride+i] + ( ((1.0f/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i]) * ((1.0f/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i]) );
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -( nn->train->ada_grad->learning_rate/(nn->train->ada_grad->delta+sqrtf(nn->train->ada_grad->costBiasDerivativeSquaredAccumulated->val[stride+i])) ) * ( ((1.0f/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i]) );
        }
        
        for (int i=0; i<n; i++) {
            nn->dense_biases->val[stride+i] = nn->dense_biases->val[stride+i] + coeff[i];
        }
        stride = stride + n;
    }
}

void rmsPropOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    nn->batch = mini_batch;
    
    init((void *)nn);
    
    miniBatchLoop((void *)nn, batch_size);
    
    // Update weights
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense_weights->shape[l][0][0];
        unsigned int n = nn->dense_weights->shape[l][1][0];
        
        float coeff[m][n];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->train->rms_prop->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)] = (nn->train->rms_prop->decayRate*nn->train->rms_prop->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)])
                + (1.0f-nn->train->rms_prop->decayRate)*( ((1.0f/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)]) * ((1.0f/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)]) );
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -( nn->train->rms_prop->learning_rate/(sqrtf(nn->train->rms_prop->delta+nn->train->rms_prop->costWeightDerivativeSquaredAccumulated->val[stride+((i*n)+j)])) ) * ( (1.0f/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)] );
            }
        }

        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense_weights->val[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                                                coeff[i][j];
            }
        }
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense_biases->shape[l][0][0];
        
        // Adpative learning rate
        float coeff[n];
        
        for (int i=0; i<n; i++) {
            nn->train->rms_prop->costBiasDerivativeSquaredAccumulated->val[stride+i] = (nn->train->rms_prop->decayRate*nn->train->rms_prop->costBiasDerivativeSquaredAccumulated->val[stride+i])
            + (1.0-nn->train->rms_prop->decayRate)*( ((1.0f/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i]) * ((1.0f/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i]) );
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -( nn->train->rms_prop->learning_rate/(sqrtf(nn->train->rms_prop->delta+nn->train->rms_prop->costBiasDerivativeSquaredAccumulated->val[stride+i])) ) * ( ((1.0f/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i]) );
        }
        
        for (int i=0; i<n; i++) {
            nn->dense_biases->val[stride+i] = nn->dense_biases->val[stride+i] + coeff[i];
        }
        stride = stride + n;
    }
}

void adamOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    nn->batch = mini_batch;
    
    init((void *)nn);
    
    miniBatchLoop((void *)nn, batch_size);
    
    // Update weights
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense_weights->shape[l][0][0];
        unsigned int n = nn->dense_weights->shape[l][1][0];
        
        float coeff[m][n];
        
        nn->train->adam->time++;
        float s_hat[m][n];
        float r_hat[m][n];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                // Update biased first moment estimate
                nn->train->adam->weightsBiasedFirstMomentEstimate->val[stride+((i*n)+j)] = (nn->train->adam->decayRate1*nn->train->adam->weightsBiasedFirstMomentEstimate->val[stride+((i*n)+j)]) + (1.0f-nn->train->adam->decayRate1)*( ((1.0f/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)]) );
                // Update biased second moment estimate
                nn->train->adam->weightsBiasedSecondMomentEstimate->val[stride+((i*n)+j)] = (nn->train->adam->decayRate2*nn->train->adam->weightsBiasedSecondMomentEstimate->val[stride+((i*n)+j)]) + (1.0f-nn->train->adam->decayRate2)*( ((1.0f/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)]) * ((1.0f/(float)batch_size)*nn->dense_costWeightDerivatives->val[stride+((i*n)+j)]) );
                
                // Correct bias in first moment
                s_hat[i][j] = nn->train->adam->weightsBiasedFirstMomentEstimate->val[stride+((i*n)+j)] / (1.0f - powf(nn->train->adam->decayRate1, (float)nn->train->adam->time));
                // Correct bias in second moment
                r_hat[i][j] = nn->train->adam->weightsBiasedSecondMomentEstimate->val[stride+((i*n)+j)] / (1.0f - powf(nn->train->adam->decayRate2, (float)nn->train->adam->time));
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -nn->train->adam->stepSize*( s_hat[i][j] / (sqrtf(r_hat[i][j]+nn->train->adam->delta)) );
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense_weights->val[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                                                coeff[i][j];
            }
        }
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int n = nn->dense_biases->shape[l][0][0];
        
        float coeff[n];
        
        float s_hat[n];
        float r_hat[n];
        for (int i=0; i<n; i++) {
            // Update biased first moment estimate
            nn->train->adam->biasesBiasedFirstMomentEstimate->val[stride+i] = (nn->train->adam->decayRate1*nn->train->adam->biasesBiasedFirstMomentEstimate->val[stride+i]) + (1.0-nn->train->adam->decayRate1)*( ((1.0f/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i]) );
            // Update biased second moment estimate
            nn->train->adam->biasesBiasedSecondMomentEstimate->val[stride+i] = (nn->train->adam->decayRate2*nn->train->adam->biasesBiasedSecondMomentEstimate->val[stride+i]) + (1.0-nn->train->adam->decayRate2)*( ((1.0f/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i]) * ((1.0f/(float)batch_size)*nn->dense_costBiasDerivatives->val[stride+i]) );
            
            // Correct bias in first moment
            s_hat[i] = nn->train->adam->biasesBiasedFirstMomentEstimate->val[stride+i] / (1.0f - powf(nn->train->adam->decayRate1, (float)nn->train->adam->time));
            // Correct bias in second moment
            r_hat[i] = nn->train->adam->biasesBiasedSecondMomentEstimate->val[stride+i] / (1.0f - powf(nn->train->adam->decayRate2, (float)nn->train->adam->time));
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -nn->train->adam->stepSize*( s_hat[i] / (sqrtf(r_hat[i]+nn->train->adam->delta)) );
        }
        
        for (int i=0; i<n; i++) {
            nn->dense_biases->val[stride+i] = nn->dense_biases->val[stride+i] + coeff[i];
        }
        stride = stride + n;
    }
}
