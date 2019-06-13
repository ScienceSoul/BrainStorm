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
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
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
        vDSP_vadd(buffer, 1, nn->dense->biases->val+stride2, 1, nn->dense->affine_transforms->val+stride2, 1, nn->dense->biases->shape[l][0][0]);
#else
        for (int i=0; i<nn->dense->biases->shape[l][0][0]; i++) {
            nn->dense->affineTransformations->val[stride2+i] = buffer[i] + nn->dense->biases->val[stride2+i];
        }
#endif
        float *vec = NULL;
        unsigned int *vec_length = NULL;
        if (nn->activation_functions_ref[l] == SOFTMAX) {
            vec = nn->dense->affine_transforms->val+stride2;
            vec_length = &(nn->dense->affine_transforms->shape[l][0][0]);
        }
        
        stride3 = stride3 + nn->dense->activations->shape[l][0][0];
        for (int i=0; i<nn->dense->activations->shape[l+1][0][0]; i++) {
            nn->dense->activations->val[stride3+i] = nn->dense->activation_functions[l](nn->dense->affine_transforms->val[stride2+i], vec, vec_length);
        }
        
        nan_to_num(nn->dense->activations->val+stride3, nn->dense->activations->shape[l+1][0][0]);
        
        stride1 = stride1 + (m * n);
        stride2 = stride2 + nn->dense->biases->shape[l][0][0];
    }
}

void backpropag_in_dense_net(void * _Nonnull neural,
                             void (* _Nullable ptr_inference_func)(void * _Nonnull self)) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    // Activations at the input layer
    int stride = nn->example_idx * nn->dense->parameters->topology[0];
    memcpy(nn->dense->activations->val,  nn->batch_inputs->val+stride, (nn->dense->parameters->topology[0])*sizeof(float));
    
    // Inference (forward pass)
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
        stride3 = stride3 + nn->dense->affine_transforms->shape[l][0][0];
    }
    
    float delta[nn->dense->parameters->max_number_nodes_in_layer];
    float buffer[nn->dense->parameters->max_number_nodes_in_layer];
    memset(delta, 0.0f, sizeof(delta));
    memset(buffer, 0.0f, sizeof(buffer));
    
    // Compute delta
    int k = (int)nn->num_channels;
    for (int i=0; i<nn->dense->activations->shape[nn->network_num_layers-1][0][0]; i++) {
        delta[i] = nn->dense->activations->val[stride2+i] - nn->batch_labels->val[nn->label_step*nn->example_idx+i];
        k++;
    }
    
    //Stride to dC/dw at last layer
    unsigned int stride4 = 0;
    unsigned int m, n;
    for (int l=0; l<nn->network_num_layers-2; l++) {
        m = nn->dense->batch_cost_weight_derivs->shape[l][0][0];
        n = nn->dense->batch_cost_weight_derivs->shape[l][1][0];
        stride4 = stride4 + (m * n);
    }
    
    stride2 = stride2 - nn->dense->activations->shape[nn->network_num_layers-2][0][0];
    n = nn->dense->batch_cost_weight_derivs->shape[nn->network_num_layers-2][1][0];
    for (int i=0; i<nn->dense->batch_cost_weight_derivs->shape[nn->network_num_layers-2][0][0]; i++) {
        for (int j=0; j<nn->dense->batch_cost_weight_derivs->shape[nn->network_num_layers-2][1][0]; j++) {
            nn->dense->batch_cost_weight_derivs->val[stride4+((i*n)+j)] = nn->dense->activations->val[stride2+j] * delta[i];
        }
    }
    memcpy(nn->dense->batch_cost_bias_derivs->val+stride3, delta, nn->dense->batch_cost_bias_derivs->shape[nn->network_num_layers-2][0][0]*sizeof(float));
    
    // The backward pass loop
    
    // Stride to weights at last layer
    stride = 0;
    for (int l=0; l<nn->network_num_layers-2; l++) {
        m = nn->dense->weights->shape[l][0][0];
        n = nn->dense->weights->shape[l][1][0];
        stride = stride + (m * n);
    }
    
    for (int l=nn->network_num_layers-2; l>0; l--) {
        stride2 = stride2 - nn->dense->activations->shape[l-1][0][0];
        stride3 = stride3 - nn->dense->affine_transforms->shape[l-1][0][0];
        stride4 = stride4 - (nn->dense->batch_cost_weight_derivs->shape[l-1][0][0]*nn->dense->batch_cost_weight_derivs->shape[l-1][1][0]);
        
        float sp[nn->dense->affine_transforms->shape[l-1][0][0]];
        for (int i=0; i<nn->dense->affine_transforms->shape[l-1][0][0]; i++) {
            sp[i] = nn->dense->activation_derivatives[l-1](nn->dense->affine_transforms->val[stride3+i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->dense->weights->shape[l][0][0], (int)nn->dense->weights->shape[l][1][0], 1.0, nn->dense->weights->val+stride, (int)nn->dense->weights->shape[l][1][0], delta, 1, 0.0, buffer, 1);
#ifdef __APPLE__
        vDSP_vmul(buffer, 1, sp, 1, delta, 1, nn->dense->affine_transforms->shape[l-1][0][0]);
#else
        for (int i=0; i<nn->dense->affineTransformations->shape[l-1][0][0]; i++) {
            delta[i] = buffer[i] * sp[i];
        }
#endif
        // dC/dw at layer l
        m = nn->dense->batch_cost_weight_derivs->shape[l-1][0][0];
        n = nn->dense->batch_cost_weight_derivs->shape[l-1][1][0];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->batch_cost_weight_derivs->val[stride4+((i*n)+j)] = nn->dense->activations->val[stride2+j] * delta[i];
            }
        }
        
        // dC/db at layer l
        memcpy(nn->dense->batch_cost_bias_derivs->val+stride3, delta, nn->dense->batch_cost_bias_derivs->shape[l-1][0][0]*sizeof(float));
        stride = stride - (nn->dense->weights->shape[l-1][0][0] * nn->dense->weights->shape[l-1][1][0]);
    }
}

void batch_accumulation_in_dense_net(void * _Nonnull neural) {
    
    // Accumulate dC/dw and dC/db
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    int offset_w = 0;
    int offset_b = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->cost_weight_derivs->shape[l][0][0];
        unsigned int n = nn->dense->cost_weight_derivs->shape[l][1][0];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->cost_weight_derivs->val[offset_w+((i*n)+j)] = nn->dense->cost_weight_derivs->val[offset_w+((i*n)+j)] + nn->dense->batch_cost_weight_derivs->val[offset_w+((i*n)+j)];
            }
        }
        for (int i=0; i<m; i++) {
            nn->dense->cost_bias_derivs->val[offset_b+i] = nn->dense->cost_bias_derivs->val[offset_b+i] + nn->dense->batch_cost_bias_derivs->val[offset_b+i];
        }
        
        offset_w = offset_w + (m * n);
        offset_b = offset_b + m;
    }
}
