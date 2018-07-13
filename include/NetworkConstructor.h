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

typedef struct scalar_dict {
    int epochs, mini_batch_size;
} scalar_dict;

typedef struct adaptive_dict {
    float scalar;
    float * _Nullable vector;
} adaptive_dict;

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
    float (* _Nullable regularizer_func)(void * _Nonnull neural, int i, int j, int n, int stride);
} regularizer_dict;

typedef struct networkConstructor {
    bool networkConstruction;
    void (* _Nullable layer)(void * _Nonnull neural, unsigned int nbNeurons, char * _Nonnull type, char * _Nullable activation, regularizer_dict * _Nullable regularizer);
    void (* _Nullable split)(void * _Nonnull neural, int n1, int n2);
    void (* _Nullable training_data)(void * _Nonnull neural, char * _Nonnull str);
    void (* _Nullable classification)(void * _Nonnull neural, int * _Nonnull vector, int n);
    void (* _Nullable scalars)(void * _Nonnull neural, scalar_dict scalars);
    void (* _Nullable adaptive_learning)(void * _Nonnull neural, char * _Nonnull method, adaptive_dict adaptive_dict);
    void * _Nonnull (* _Nullable optimizer)(void * _Nonnull neural, optimizer_dict optimizer_dict);
    
} networkConstructor;

networkConstructor * _Nonnull allocateConstructor(void);

#endif /* NetworkConstructor_h */
