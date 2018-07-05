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

struct weightMatrixDimension {
    unsigned int m, n;
};

struct biasVectorDimension {
    unsigned int n;
};

struct parameters {
    unsigned int gridDimension;
    unsigned int numberOfLayers;
    unsigned int numberOfFeatures;
    unsigned int numberOfOutputs;
    
    weightMatrixDimension weightsDim[100];
    biasVectorDimension biasesDim[100];
};

inline void matrixVectorMul(device float *a, device float *x, uint m, uint n, uint gridID, uint gridDimension) {
    
    float buffer[BUFFER_MAX_LENGTH];
    uint idx = 0;
    for(uint i=0; i<m; i++) {
        float sum = 0.0f;
        for(uint j=0; j<n; j++) {
            sum = fma(a[idx], x[gridID+(j*gridDimension)], sum);
            idx++;
        }
        buffer[i] = sum;
    }
    
    for(uint i=0; i<m; i++) {
        x[gridID+(i*gridDimension)] = buffer[i];
    }
}

inline float sigmoid(float z) {
    return 1.0f / (1.0f + exp(-z));
}

inline float nanToNum (float val) {
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
                        device float *groundTruth [[ buffer(4) ]],
                        constant parameters &params [[ buffer(5) ]],
                        uint grid_id [[ thread_position_in_grid ]],
                        uint group_id [[ thread_position_in_threadgroup ]],
                        uint threads_per_threadgroup [[ threads_per_threadgroup ]]) {
    
    if (grid_id >= params.gridDimension) return;
    
    for(uint i=0; i<params.numberOfFeatures; i++) {
        activations[grid_id+(i*params.gridDimension)] = data[grid_id+(i*params.gridDimension)];
    }
    
//    for(uint i=0; i<threads_per_threadgroup; i++) {
//        for(uint j=0; j<params.numberOfFeatures; j++) {
//            workBuffer[group_id+(j*threads_per_threadgroup)] = data[grid_id+(j*params.gridDimension)];
//        }
//    }
//    threadgroup_barrier(mem_flags::mem_device);
    
    uint stride1 = 0;
    uint stride2 = 0;
    for(uint l=0; l<params.numberOfLayers-1; l++) {
        uint m = params.weightsDim[l].m;
        uint n = params.weightsDim[l].n;
        
        // Wa
        matrixVectorMul(weights+stride1, activations, m, n, grid_id, params.gridDimension);
        
        // z = Wa + b
         for (uint i=0; i<m; i++) {
             activations[grid_id+(i*params.gridDimension)] = activations[grid_id+(i*params.gridDimension)] + biases[stride2+i];
         }
        // sigmoid(z)
        for (uint i=0; i<m; i++) {
            activations[grid_id+(i*params.gridDimension)] = sigmoid(activations[grid_id+(i*params.gridDimension)]);
            activations[grid_id+(i*params.gridDimension)] = nanToNum(activations[grid_id+(i*params.gridDimension)]);
        }
        stride1 = stride1 + (m * n);
        stride2 = stride2 + params.biasesDim[l].n;
    }
    
    uint idx = 0;
    float max = -HUGE_VALF;
    for (uint j=0; j<params.numberOfOutputs; j++) {
        if (activations[grid_id+(j*params.gridDimension)] > max) {
            max = activations[grid_id+(j*params.gridDimension)];
            idx = j;
        }
    }
    if (idx == groundTruth[grid_id]) {
        groundTruth[grid_id] = 1.0f;
    } else groundTruth[grid_id] = 0.0f;
}
