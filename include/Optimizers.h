//
//  Optimization.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 28/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef Optimization_h
#define Optimization_h

#include "NetworkUtils.h"

typedef struct dense {
    tensor * _Nullable costWeightDerivativeSquaredAccumulated;
    tensor * _Nullable costBiasDerivativeSquaredAccumulated;
    
    tensor * _Nullable weightsBiasedFirstMomentEstimate;
    tensor * _Nullable weightsBiasedSecondMomentEstimate;
    tensor * _Nullable biasesBiasedFirstMomentEstimate;
    tensor * _Nullable biasesBiasedSecondMomentEstimate;
} dense;

typedef struct conv2d {
    tensor * _Nullable costWeightDerivativeSquaredAccumulated;
    tensor * _Nullable costBiasDerivativeSquaredAccumulated;
    
    tensor * _Nullable weightsBiasedFirstMomentEstimate;
    tensor * _Nullable weightsBiasedSecondMomentEstimate;
    tensor * _Nullable biasesBiasedFirstMomentEstimate;
    tensor * _Nullable biasesBiasedSecondMomentEstimate;
} conv2d;

typedef struct GradientDescentOptimizer {
    float learning_rate;
    void (* _Nullable minimize)(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
} GradientDescentOptimizer;

typedef struct MomentumOptimizer {
    float learning_rate;
    float momentum_coefficient;
    void (* _Nullable minimize)(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
} MomentumOptimizer;

typedef struct AdaGradOptimizer {
    float learning_rate;
    float delta;
    dense * _Nullable dense;
    conv2d * _Nullable conv2d;
    void (* _Nullable minimize)(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
} AdaGradOptimizer;

typedef struct RMSPropOptimizer {
    float learning_rate;
    float delta;
    float decay_rate;
    dense * _Nullable dense;
    conv2d * _Nullable conv2d;
    void (* _Nullable minimize)(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
} RMSPropOptimizer;

typedef struct AdamOptimizer {
    unsigned int time;
    float step_size;
    float delta;
    float decay_rate1;
    float decay_rate2;
    dense * _Nullable dense;
    conv2d * _Nullable conv2d;
    void (* _Nullable minimize)(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
} AdamOptimizer;

void gradientDescentOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
void momentumOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
void adaGradOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
void rmsPropOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
void adamOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);

#endif /* Optimization_h */
