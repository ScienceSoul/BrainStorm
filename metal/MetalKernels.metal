//
//  MetalKernels.metal
//  FeedforwardNT
//
//  Hakime Seddik on 11/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

#define BUFFER_MAX_LENGTH 1000

struct weight_matrix_dimension {
    unsigned int m, n;
};

struct bias_vector_dimension {
    unsigned int n;
};

struct parameters {
    unsigned int grid_dim;
    unsigned int num_layers;
    unsigned int num_features;
    unsigned int num_outputs;
    
    weight_matrix_dimension weights_dim[100];
    bias_vector_dimension biases_dim[100];
};

inline void mat_vec_mul(device float *a, device float *x, uint m, uint n, uint grid_id, uint grid_dim) {
    
    float buffer[BUFFER_MAX_LENGTH];
    uint idx = 0;
    for(uint i=0; i<m; i++) {
        float sum = 0.0f;
        for(uint j=0; j<n; j++) {
            sum = fma(a[idx], x[grid_id+(j*grid_dim)], sum);
            idx++;
        }
        buffer[i] = sum;
    }
    
    for(uint i=0; i<m; i++) {
        x[grid_id+(i*grid_dim)] = buffer[i];
    }
}

inline float sigmoid(float z) {
    return 1.0f / (1.0f + exp(-z));
}

inline float nan_to_num(float val) {
    float number = val;
    if (isnan(val) != 0) number= 0.0f;
    if (isinf(val) != 0) {
        if (val > 0) {
            number = HUGE_VALF;
        } else if (val < 0) {
            number = -HUGE_VALF;
        }
    }
    return number;
}

kernel void feedforward(device float *data [[ buffer(0) ]],
                        device float *weights [[ buffer(1) ]],
                        device float *biases [[ buffer(2) ]],
                        device float *activations [[ buffer(3) ]],
                        device float *ground_truth [[ buffer(4) ]],
                        constant parameters &params [[ buffer(5) ]],
                        uint grid_id [[ thread_position_in_grid ]],
                        uint group_id [[ thread_position_in_threadgroup ]],
                        uint threads_per_threadgroup [[ threads_per_threadgroup ]]) {
    
    if (grid_id >= params.grid_dim) return;
    
    for(uint i=0; i<params.num_features; i++) {
        activations[grid_id+(i*params.grid_dim)] = data[grid_id+(i*params.grid_dim)];
    }
    
//    for(uint i=0; i<threads_per_threadgroup; i++) {
//        for(uint j=0; j<params.num_features; j++) {
//            workBuffer[group_id+(j*threads_per_threadgroup)] = data[grid_id+(j*params.grid_dim)];
//        }
//    }
//    threadgroup_barrier(mem_flags::mem_device);
    
    uint stride1 = 0;
    uint stride2 = 0;
    for(uint l=0; l<params.num_layers-1; l++) {
        uint m = params.weights_dim[l].m;
        uint n = params.weights_dim[l].n;
        
        // Wa
        mat_vec_mul(weights+stride1, activations, m, n, grid_id, params.grid_dim);
        
        // z = Wa + b
         for (uint i=0; i<m; i++) {
             activations[grid_id+(i*params.grid_dim)] = activations[grid_id+(i*params.grid_dim)] + biases[stride2+i];
         }
        // sigmoid(z)
        for (uint i=0; i<m; i++) {
            activations[grid_id+(i*params.grid_dim)] = sigmoid(activations[grid_id+(i*params.grid_dim)]);
            activations[grid_id+(i*params.grid_dim)] = nan_to_num(activations[grid_id+(i*params.grid_dim)]);
        }
        stride1 = stride1 + (m * n);
        stride2 = stride2 + params.biases_dim[l].n;
    }
    
    uint idx = 0;
    float max = -HUGE_VALF;
    for (uint j=0; j<params.num_outputs; j++) {
        if (activations[grid_id+(j*params.grid_dim)] > max) {
            max = activations[grid_id+(j*params.grid_dim)];
            idx = j;
        }
    }
    if (idx == ground_truth[grid_id]) {
        ground_truth[grid_id] = 1.0f;
    } else ground_truth[grid_id] = 0.0f;
}
