//
//  NetworkUtils.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 26/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef NetworkUtils_h
#define NetworkUtils_h

#include <stdbool.h>
#include "Utils.h"

typedef enum activation_functions {
    SIGMOID=1,            // Logistic sigmoid
    RELU,                 // Rectified linear unit
    LEAKY_RELU,           // Leaky ReLU
    ELU,                  // Exponential linear unit
    TANH,                 // Hyperbolic tangent
    SOFTMAX,              // Softmax
    CUSTOM
} activation_functions;

typedef enum convlution_layer_types {
    FEED=1,
    CONVOLUTION,
    POOLING,
    FULLY_CONNECTED,
} convlution_layers_type;

typedef enum conv2d_padding {
    VALID=1,
    SAME
} conv2d_padding;

typedef enum pooling_ops {
    MAX_POOLING=1,
    L2_POOLING,
    AVERAGE_POOLING
} pooling_ops;

typedef struct tensor {
    unsigned int shape[MAX_NUMBER_NETWORK_LAYERS][MAX_TENSOR_RANK][1];
    unsigned int rank;
    float * _Nullable val;
    int * _Nullable int32_val;
} tensor;

typedef struct tensor_dict {
    bool init_weights;
    bool init_with_value;
    bool full_connected;
    unsigned int flattening_length;
    unsigned int shape[MAX_NUMBER_NETWORK_LAYERS][MAX_TENSOR_RANK][1];
    unsigned int rank;
    float init_value;
} tensor_dict;

void variance_scaling_initializer(void * _Nonnull object, float * _Nullable factor, char * _Nullable mode, bool * _Nullable uniform, int layer, int offset, float * _Nullable val);

void xavier_initializer(void * _Nonnull object, float * _Nullable factor, char * _Nullable mode, bool * _Nullable uniform, int layer, int offset, float * _Nullable val);

void random_normal_initializer(void * _Nonnull object, float * _Nullable factor, char * _Nullable mode, bool * _Nullable uniform, int layer, int offset, float * _Nullable val);

void value_initializer(void * _Nonnull object, float * _Nullable factor, char * _Nullable mode, bool * _Nullable uniform, int layer, int offset, float * _Nullable val);

void * _Nullable tensor_create(void * _Nullable self, tensor_dict tensor_dict);

tensor_dict * _Nonnull init_tensor_dict(void);

void __attribute__((overloadable))shuffle(void * _Nonnull features, void * _Nullable labels, int num_classifications, int * _Nullable num_features);

int loadParametersFromImputFile(void * _Nonnull self, const char * _Nonnull paraFile);

#endif /* NetworkUtils_h */
