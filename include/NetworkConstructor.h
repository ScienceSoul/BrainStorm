//
//  NetworkConstructor.h
//  BrainStorm
//
//  Created by Hakime Seddik on 06/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef NetworkConstructor_h
#define NetworkConstructor_h

#include <stdbool.h>
#include "NetworkUtils.h"

typedef struct scalar_dict {
    int epochs, mini_batch_size;
} scalar_dict;

typedef struct layer_dict {
    unsigned int num_neurons;
    unsigned int filters;
    unsigned int kernel_size[2];
    unsigned int strides[2];
    conv2d_padding padding;
    activation_functions activation;
    pooling_ops pooling_op;
    void (* _Nullable kernel_initializer)(void * _Nonnull neural, void * _Nonnull kernel, int l, int offset);
} layer_dict;

typedef struct optimizer_dict {
    char * _Nullable optimizer;
    float learning_rate;
    float momentum;
    float delta;
    float decay_rate1;
    float decay_rate2;
    float step_size;
    float * _Nullable vector;
} optimizer_dict;

typedef struct regularizer_dict {
    float regularization_factor;
    float (* _Nullable regularizer_func)(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2);
} regularizer_dict;

typedef struct networkConstructor {
    bool networkConstruction;
    void (* _Nullable feed)(void * _Nonnull neural, unsigned int shape[_Nonnull 3], unsigned int dimension, unsigned int * _Nullable num_channels);
    void (* _Nullable layer_dense)(void * _Nonnull neural, layer_dict layer_dict, regularizer_dict * _Nullable regularizer);
    void (* _Nullable layer_conv2d)(void * _Nonnull neural, layer_dict layer_dict, regularizer_dict * _Nullable regularizer);
    void (* _Nullable layer_pool)(void * _Nonnull neural, layer_dict layer_dict);
    
    void (* _Nullable split)(void * _Nonnull neural, int n1, int n2);
    void (* _Nullable training_data)(void * _Nonnull neural, char * _Nonnull str);
    void (* _Nullable classification)(void * _Nonnull neural, int * _Nonnull vector, int n);
    void (* _Nullable scalars)(void * _Nonnull neural, scalar_dict scalars);
    void * _Nonnull (* _Nullable optimizer)(void * _Nonnull neural, optimizer_dict optimizer_dict);
    
} networkConstructor;

networkConstructor * _Nonnull allocateConstructor(void);

#endif /* NetworkConstructor_h */
