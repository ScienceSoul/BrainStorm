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
    
    costWeightDerivativeNode *dcdwNodePt = nn->networkCostWeightDerivatives;
    while (dcdwNodePt != NULL) {
        memset(*dcdwNodePt->dcdw, 0.0f, (dcdwNodePt->m*dcdwNodePt->n)*sizeof(float));
        dcdwNodePt = dcdwNodePt->next;
    }
    
    costBiaseDerivativeNode *dcdbNodePt = nn->networkCostBiaseDerivatives;
    while (dcdbNodePt != NULL) {
        memset(dcdbNodePt->dcdb, 0.0f, dcdbNodePt->n*sizeof(float));
        dcdbNodePt = dcdbNodePt->next;
    }
    
    costWeightDerivativeNode *delta_dcdwNodePt = nn->deltaNetworkCostWeightDerivatives;
    while (delta_dcdwNodePt != NULL) {
        memset(*delta_dcdwNodePt->dcdw, 0.0f, (delta_dcdwNodePt->m*delta_dcdwNodePt->n)*sizeof(float));
        delta_dcdwNodePt = delta_dcdwNodePt->next;
    }
    
    costBiaseDerivativeNode *delta_dcdbNodePt = nn->deltaNetworkCostBiaseDerivatives;
    while (delta_dcdbNodePt != NULL) {
        memset(delta_dcdbNodePt->dcdb, 0.0f, delta_dcdbNodePt->n*sizeof(float));
        delta_dcdbNodePt = delta_dcdbNodePt->next;
    }
}

void gradientDescentOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    nn->batch = mini_batch;
    
    init((void *)nn);
    
    miniBatchLoop((void *)nn, batch_size);
    
    // Update weights
    unsigned int stride = 0;
    unsigned int l = 0;
    costWeightDerivativeNode *dcdwNodePt = nn->networkCostWeightDerivatives;
    while (dcdwNodePt != NULL) {
        unsigned int m = nn->weightsDimensions[l].m;
        unsigned int n = nn->weightsDimensions[l].n;
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->weights[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) -
                     (nn->train->gradient_descent->learning_rate/(float)batch_size)*dcdwNodePt->dcdw[i][j];
            }
        }
        dcdwNodePt = dcdwNodePt->next;
        l++;
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    l = 0;
    costBiaseDerivativeNode *dcdbNodePt = nn->networkCostBiaseDerivatives;
    while (dcdbNodePt != NULL) {
        unsigned int n = nn->biasesDimensions[l].n;
        
        for (int i=0; i<n; i++) {
            nn->biases[stride+i] = nn->biases[stride+i] - (nn->train->gradient_descent->learning_rate/(float)batch_size)*dcdbNodePt->dcdb[i];
        }
        dcdbNodePt = dcdbNodePt->next;
        l++;
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
    unsigned int l = 0;
    costWeightDerivativeNode *dcdwNodePt = nn->networkCostWeightDerivatives;
    while (dcdwNodePt != NULL) {
        unsigned int m = nn->weightsDimensions[l].m;
        unsigned int n = nn->weightsDimensions[l].n;
        
        float coeff[m][n];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = (nn->train->momentum->learning_rate/(float)batch_size)*dcdwNodePt->dcdw[i][j];
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->weightsVelocity[stride+((i*n)+j)] = nn->train->momentum->momentum_coefficient *
                                                 nn->weightsVelocity[stride+((i*n)+j)] - coeff[i][j];
                nn->weights[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                                                nn->weightsVelocity[stride+((i*n)+j)];
            }
        }
        dcdwNodePt = dcdwNodePt->next;
        l++;
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    l = 0;
    costBiaseDerivativeNode *dcdbNodePt = nn->networkCostBiaseDerivatives;
    while (dcdbNodePt != NULL) {
        unsigned int n = nn->biasesDimensions[l].n;
        
        // Adpative learning rate
        float coeff[n];
        
        for (int i=0; i<n; i++) {
            coeff[i] = (nn->train->momentum->learning_rate/(float)batch_size)*dcdbNodePt->dcdb[i];
        }
        
        for (int i=0; i<n; i++) {
            nn->biasesVelocity[stride+i] = nn->train->momentum->momentum_coefficient *
                                                nn->biasesVelocity[stride+i] - coeff[i];
            nn->biases[stride+i] = nn->biases[stride+i] + nn->biasesVelocity[stride+i];
        }
        dcdbNodePt = dcdbNodePt->next;
        l++;
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
    unsigned int l = 0;
    costWeightDerivativeNode *dcdwNodePt = nn->networkCostWeightDerivatives;
    while (dcdwNodePt != NULL) {
        unsigned int m = nn->weightsDimensions[l].m;
        unsigned int n = nn->weightsDimensions[l].n;
        
        float coeff[m][n];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->train->ada_grad->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)] = nn->train->ada_grad->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)] + ( ((1.0f/(float)batch_size)*dcdwNodePt->dcdw[i][j]) * ((1.0f/(float)batch_size)*dcdwNodePt->dcdw[i][j]) );
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -( nn->train->ada_grad->learning_rate/(nn->train->ada_grad->delta+sqrtf(nn->train->ada_grad->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)])) ) * ( (1.0f/(float)batch_size)*dcdwNodePt->dcdw[i][j] );
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->weights[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                                                coeff[i][j];
            }
        }
        dcdwNodePt = dcdwNodePt->next;
        l++;
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    l = 0;
    costBiaseDerivativeNode *dcdbNodePt = nn->networkCostBiaseDerivatives;
    while (dcdbNodePt != NULL) {
        unsigned int n = nn->biasesDimensions[l].n;

        float coeff[n];
        
        for (int i=0; i<n; i++) {
            nn->train->ada_grad->costBiasDerivativeSquaredAccumulated[stride+i] = nn->train->ada_grad->costBiasDerivativeSquaredAccumulated[stride+i] + ( ((1.0f/(float)batch_size)*dcdbNodePt->dcdb[i]) * ((1.0f/(float)batch_size)*dcdbNodePt->dcdb[i]) );
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -( nn->train->ada_grad->learning_rate/(nn->train->ada_grad->delta+sqrtf(nn->train->ada_grad->costBiasDerivativeSquaredAccumulated[stride+i])) ) * ( ((1.0f/(float)batch_size)*dcdbNodePt->dcdb[i]) );
        }
        
        for (int i=0; i<n; i++) {
            nn->biases[stride+i] = nn->biases[stride+i] + coeff[i];
        }
        dcdbNodePt = dcdbNodePt->next;
        l++;
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
    unsigned int l = 0;
    costWeightDerivativeNode *dcdwNodePt = nn->networkCostWeightDerivatives;
    while (dcdwNodePt != NULL) {
        unsigned int m = nn->weightsDimensions[l].m;
        unsigned int n = nn->weightsDimensions[l].n;
        
        float coeff[m][n];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->train->rms_prop->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)] = (nn->train->rms_prop->decayRate*nn->train->rms_prop->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)])
                + (1.0f-nn->train->rms_prop->decayRate)*( ((1.0f/(float)batch_size)*dcdwNodePt->dcdw[i][j]) * ((1.0f/(float)batch_size)*dcdwNodePt->dcdw[i][j]) );
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -( nn->train->rms_prop->learning_rate/(sqrtf(nn->train->rms_prop->delta+nn->train->rms_prop->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)])) ) * ( (1.0f/(float)batch_size)*dcdwNodePt->dcdw[i][j] );
            }
        }

        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->weights[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                                                coeff[i][j];
            }
        }
        dcdwNodePt = dcdwNodePt->next;
        l++;
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    l = 0;
    costBiaseDerivativeNode *dcdbNodePt = nn->networkCostBiaseDerivatives;
    while (dcdbNodePt != NULL) {
        unsigned int n = nn->biasesDimensions[l].n;
        
        // Adpative learning rate
        float coeff[n];
        
        for (int i=0; i<n; i++) {
            nn->train->rms_prop->costBiasDerivativeSquaredAccumulated[stride+i] = (nn->train->rms_prop->decayRate*nn->train->rms_prop->costBiasDerivativeSquaredAccumulated[stride+i])
            + (1.0-nn->train->rms_prop->decayRate)*( ((1.0f/(float)batch_size)*dcdbNodePt->dcdb[i]) * ((1.0f/(float)batch_size)*dcdbNodePt->dcdb[i]) );
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -( nn->train->rms_prop->learning_rate/(sqrtf(nn->train->rms_prop->delta+nn->train->rms_prop->costBiasDerivativeSquaredAccumulated[stride+i])) ) * ( ((1.0f/(float)batch_size)*dcdbNodePt->dcdb[i]) );
        }
        
        for (int i=0; i<n; i++) {
            nn->biases[stride+i] = nn->biases[stride+i] + coeff[i];
        }
        dcdbNodePt = dcdbNodePt->next;
        l++;
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
    unsigned int l = 0;
    costWeightDerivativeNode *dcdwNodePt = nn->networkCostWeightDerivatives;
    while (dcdwNodePt != NULL) {
        unsigned int m = nn->weightsDimensions[l].m;
        unsigned int n = nn->weightsDimensions[l].n;
        
        float coeff[m][n];
        
        nn->train->adam->time++;
        float s_hat[m][n];
        float r_hat[m][n];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                // Update biased first moment estimate
                nn->train->adam->weightsBiasedFirstMomentEstimate[stride+((i*n)+j)] = (nn->train->adam->decayRate1*nn->train->adam->weightsBiasedFirstMomentEstimate[stride+((i*n)+j)]) + (1.0f-nn->train->adam->decayRate1)*( ((1.0f/(float)batch_size)*dcdwNodePt->dcdw[i][j]) );
                // Update biased second moment estimate
                nn->train->adam->weightsBiasedSecondMomentEstimate[stride+((i*n)+j)] = (nn->train->adam->decayRate2*nn->train->adam->weightsBiasedSecondMomentEstimate[stride+((i*n)+j)]) + (1.0f-nn->train->adam->decayRate2)*( ((1.0f/(float)batch_size)*dcdwNodePt->dcdw[i][j]) * ((1.0f/(float)batch_size)*dcdwNodePt->dcdw[i][j]) );
                
                // Correct bias in first moment
                s_hat[i][j] = nn->train->adam->weightsBiasedFirstMomentEstimate[stride+((i*n)+j)] / (1.0f - powf(nn->train->adam->decayRate1, (float)nn->train->adam->time));
                // Correct bias in second moment
                r_hat[i][j] = nn->train->adam->weightsBiasedSecondMomentEstimate[stride+((i*n)+j)] / (1.0f - powf(nn->train->adam->decayRate2, (float)nn->train->adam->time));
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                coeff[i][j] = -nn->train->adam->stepSize*( s_hat[i][j] / (sqrtf(r_hat[i][j]+nn->train->adam->delta)) );
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->weights[stride+((i*n)+j)] = nn->regularizer[l]((void *)nn, i, j, n, stride) +
                                                coeff[i][j];
            }
        }
        dcdwNodePt = dcdwNodePt->next;
        l++;
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    l = 0;
    costBiaseDerivativeNode *dcdbNodePt = nn->networkCostBiaseDerivatives;
    while (dcdbNodePt != NULL) {
        unsigned int n = nn->biasesDimensions[l].n;
        
        float coeff[n];
        
        float s_hat[n];
        float r_hat[n];
        for (int i=0; i<n; i++) {
            // Update biased first moment estimate
            nn->train->adam->biasesBiasedFirstMomentEstimate[stride+i] = (nn->train->adam->decayRate1*nn->train->adam->biasesBiasedFirstMomentEstimate[stride+i]) + (1.0-nn->train->adam->decayRate1)*( ((1.0f/(float)batch_size)*dcdbNodePt->dcdb[i]) );
            // Update biased second moment estimate
            nn->train->adam->biasesBiasedSecondMomentEstimate[stride+i] = (nn->train->adam->decayRate2*nn->train->adam->biasesBiasedSecondMomentEstimate[stride+i]) + (1.0-nn->train->adam->decayRate2)*( ((1.0f/(float)batch_size)*dcdbNodePt->dcdb[i]) * ((1.0f/(float)batch_size)*dcdbNodePt->dcdb[i]) );
            
            // Correct bias in first moment
            s_hat[i] = nn->train->adam->biasesBiasedFirstMomentEstimate[stride+i] / (1.0f - powf(nn->train->adam->decayRate1, (float)nn->train->adam->time));
            // Correct bias in second moment
            r_hat[i] = nn->train->adam->biasesBiasedSecondMomentEstimate[stride+i] / (1.0f - powf(nn->train->adam->decayRate2, (float)nn->train->adam->time));
        }
        for (int i=0; i<n; i++) {
            coeff[i] = -nn->train->adam->stepSize*( s_hat[i] / (sqrtf(r_hat[i]+nn->train->adam->delta)) );
        }
        
        for (int i=0; i<n; i++) {
            nn->biases[stride+i] = nn->biases[stride+i] + coeff[i];
        }
        dcdbNodePt = dcdbNodePt->next;
        l++;
        stride = stride + n;
    }
}
