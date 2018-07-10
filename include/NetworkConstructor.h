//
//  NetworkConstructor.h
//  BrainStorm
//
//  Created by Hakime Seddik on 06/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef NetworkConstructor_h
#define NetworkConstructor_h


typedef struct scalar_dict {
    int epochs, mini_batch_size;
    float learning_rate;
    float regularization_factor;
    float momentum_coefficient;
} scalar_dict;

typedef struct adaptive_dict {
    float scalar;
    float * _Nullable vector;
} adaptive_dict;

typedef struct networkConstructor {
    bool networkConstruction;
    void (* _Nullable layer)(void * _Nonnull neural, unsigned int nbNeurons, char * _Nonnull type, char * _Nullable activation);
    void (* _Nullable split)(void * _Nonnull neural, int n1, int n2);
    void (* _Nullable training_data)(void * _Nonnull neural, char * _Nonnull str);
    void (* _Nullable classification)(void * _Nonnull neural, int * _Nonnull vector, int n);
    void (* _Nullable scalars)(void * _Nonnull neural, scalar_dict scalars);
    void (* _Nullable adaptive_learning)(void * _Nonnull neural, char * _Nonnull method, adaptive_dict adaptive_dict);
} networkConstructor;

networkConstructor * _Nonnull allocateConstructor(void);

#endif /* NetworkConstructor_h */
