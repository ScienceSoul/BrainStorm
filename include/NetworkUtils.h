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
    SOFTMAX               // Softmax
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
    float * _Nullable val;
    unsigned int shape[MAX_NUMBER_NETWORK_LAYERS][MAX_TENSOR_RANK][1];
    unsigned int rank;
} tensor;

typedef struct tensor_dict {
    bool init_neural_params;
    bool init_with_value;
    bool full_connected;
    unsigned int flattening_length;
    unsigned int shape[MAX_NUMBER_NETWORK_LAYERS][MAX_TENSOR_RANK][1];
    unsigned int rank;
    float init_value;
} tensor_dict;

void value_initializer(void * _Nonnull neural, void * _Nonnull object, int l, int offset, float * _Nullable val);
void standard_normal_initializer(void * _Nonnull neural, void * _Nonnull object, int l, int offset);
void xavier_he_initializer(void * _Nonnull neural, void * _Nonnull object, int l, int offset);

void * _Nullable tensor_create(void * _Nonnull self, tensor_dict tensor_dict);

int loadParametersFromImputFile(void * _Nonnull self, const char * _Nonnull paraFile);

tensor_dict * _Nonnull init_tensor_dict(void);

#endif /* NetworkUtils_h */
