//
//  DenseNetOps.c
//  BrainStorm
//
//  Created by Hakime Seddik on 13/08/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include "cblas.h"
#endif

#include "NeuralNetwork.h"
#include "DenseNetOps.h"

void inference_in_dense_net(void * _Nonnull neural) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    unsigned int stride1 = 0;
    unsigned int stride2 = 0;
    unsigned int stride3 = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        float buffer[nn->dense->activations->shape[l+1][0][0]];
        memset(buffer, 0.0f, sizeof(buffer));
        
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n, 1.0, nn->dense->weights->val+stride1, (int)n, nn->dense->activations->val+stride3, 1, 0.0, buffer, 1);
#ifdef __APPLE__
        vDSP_vadd(buffer, 1, nn->dense->biases->val+stride2, 1, nn->dense->affineTransformations->val+stride2, 1, nn->dense->biases->shape[l][0][0]);
#else
        for (int i=0; i<nn->biasesDimensions[l].n; i++) {
            nn->dense_affineTransformations->val[stride2+i] = buffer[i] + nn->biases[stride2+i];
        }
#endif
        float *vec = NULL;
        unsigned int *vec_length = NULL;
        if (nn->activationFunctionsRef[l] == SOFTMAX) {
            vec = nn->dense->affineTransformations->val+stride2;
            vec_length = &(nn->dense->affineTransformations->shape[l][0][0]);
        }
        
        stride3 = stride3 + nn->dense->activations->shape[l][0][0];
        for (int i=0; i<nn->dense->activations->shape[l+1][0][0]; i++) {
            nn->dense->activations->val[stride3+i] = nn->dense->activationFunctions[l](nn->dense->affineTransformations->val[stride2+i], vec, vec_length);
        }
        
        nanToNum(nn->dense->activations->val+stride3, nn->dense->activations->shape[l+1][0][0]);
        
        stride1 = stride1 + (m * n);
        stride2 = stride2 + nn->dense->biases->shape[l][0][0];
    }
}

void backpropag_in_dense_net(void * _Nonnull neural,
                             void (* _Nullable ptr_inference_func)(void * _Nonnull self)) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    // Activations at the input layer
    for (int i=0; i<nn->num_channels; i++) {
        nn->dense->activations->val[i] = nn->batch[nn->example_idx][i];
    }
    
    // Feedforward
    ptr_inference_func(nn);
    
    // ------------- Backward pass
    // At last layer
    
    // Stride to activations at last layer
    unsigned stride2 = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        stride2 = stride2 + nn->dense->activations->shape[l][0][0];
    }
    
    // Stride to affine transformations and dc/db at last layer
    unsigned stride3 = 0;
    for (int l=0; l<nn->network_num_layers-2; l++) {
        stride3 = stride3 + nn->dense->affineTransformations->shape[l][0][0];
    }
    
    float delta[nn->dense->parameters->max_number_of_nodes_in_layer];
    float buffer[nn->dense->parameters->max_number_of_nodes_in_layer];
    memset(delta, 0.0f, sizeof(delta));
    memset(buffer, 0.0f, sizeof(buffer));
    
    // Compute delta
    int k = (int)nn->num_channels;
    for (int i=0; i<nn->dense->activations->shape[nn->network_num_layers-1][0][0]; i++) {
        delta[i] = nn->dense->activations->val[stride2+i] - nn->batch[nn->example_idx][k];
        k++;
    }
    
    //Stride to dc/dw at last layer
    unsigned int stride4 = 0;
    unsigned int m, n;
    for (int l=0; l<nn->network_num_layers-2; l++) {
        m = nn->dense->batchCostWeightDeriv->shape[l][0][0];
        n = nn->dense->batchCostWeightDeriv->shape[l][1][0];
        stride4 = stride4 + (m * n);
    }
    
    stride2 = stride2 - nn->dense->activations->shape[nn->network_num_layers-2][0][0];
    n = nn->dense->batchCostWeightDeriv->shape[nn->network_num_layers-2][1][0];
    for (int i=0; i<nn->dense->batchCostWeightDeriv->shape[nn->network_num_layers-2][0][0]; i++) {
        for (int j=0; j<nn->dense->batchCostWeightDeriv->shape[nn->network_num_layers-2][1][0]; j++) {
            nn->dense->batchCostWeightDeriv->val[stride4+((i*n)+j)] = nn->dense->activations->val[stride2+j] * delta[i];
        }
    }
    for (int i=0; i<nn->dense->batchCostBiasDeriv->shape[nn->network_num_layers-2][0][0]; i++) {
        nn->dense->batchCostBiasDeriv->val[stride3+i] = delta[i];
    }
    
    // The backward pass loop
    
    // Stride to weights at last layer
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-2; l++) {
        m = nn->dense->weights->shape[l][0][0];
        n = nn->dense->weights->shape[l][1][0];
        stride = stride + (m * n);
    }
    
    for (int l=nn->network_num_layers-2; l>0; l--) {
        stride2 = stride2 - nn->dense->activations->shape[l-1][0][0];
        stride3 = stride3 - nn->dense->affineTransformations->shape[l-1][0][0];
        stride4 = stride4 - (nn->dense->batchCostWeightDeriv->shape[l-1][0][0]*nn->dense->batchCostWeightDeriv->shape[l-1][1][0]);
        
        float sp[nn->dense->affineTransformations->shape[l-1][0][0]];
        for (int i=0; i<nn->dense->affineTransformations->shape[l-1][0][0]; i++) {
            sp[i] = nn->dense->activationDerivatives[l-1](nn->dense->affineTransformations->val[stride3+i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->dense->weights->shape[l][0][0], (int)nn->dense->weights->shape[l][1][0], 1.0, nn->dense->weights->val+stride, (int)nn->dense->weights->shape[l][1][0], delta, 1, 0.0, buffer, 1);
        for (int i=0; i<nn->dense->affineTransformations->shape[l-1][0][0]; i++) {
            delta[i] = buffer[i] * sp[i];
        }
        // dc/dw at layer l
        m = nn->dense->batchCostWeightDeriv->shape[l-1][0][0];
        n = nn->dense->batchCostWeightDeriv->shape[l-1][1][0];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->batchCostWeightDeriv->val[stride4+((i*n)+j)] = nn->dense->activations->val[stride2+j] * delta[i];
            }
        }
        // dc/db at layer l
        for (int i=0; i<nn->dense->batchCostBiasDeriv->shape[l-1][0][0]; i++) {
            nn->dense->batchCostBiasDeriv->val[stride3+i] = delta[i];
        }
        
        stride = stride - (nn->dense->weights->shape[l-1][0][0] * nn->dense->weights->shape[l-1][1][0]);
    }
}

void batch_accumulation_in_dense_net(void * _Nonnull neural) {
    
    // Accumulate dcdw and dc/db
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    unsigned stride1 = 0;
    unsigned stride2 = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->costWeightDerivatives->shape[l][0][0];
        unsigned int n = nn->dense->costWeightDerivatives->shape[l][1][0];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->costWeightDerivatives->val[stride1+((i*n)+j)] = nn->dense->costWeightDerivatives->val[stride1+((i*n)+j)] + nn->dense->batchCostWeightDeriv->val[stride1+((i*n)+j)];
            }
        }
        for (int i=0; i<m; i++) {
            nn->dense->costBiasDerivatives->val[stride2+i] = nn->dense->costBiasDerivatives->val[stride2+i] + nn->dense->batchCostBiasDeriv->val[stride2+i];
        }
        
        stride1 = stride1 + (m * n);
        stride2 = stride2 + m;
    }
}
