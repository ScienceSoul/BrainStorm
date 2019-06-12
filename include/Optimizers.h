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
    tensor * _Nullable cost_weight_derivative_squared_accumulated;
    tensor * _Nullable cost_bias_derivative_squared_accumulated;
    
    tensor * _Nullable weights_biased_first_moment_estimate;
    tensor * _Nullable weights_biased_second_moment_estimate;
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

typedef struct gradient_descent_optimizer {
    float learning_rate;
    void (* _Nullable minimize)(void * _Nonnull neural, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
} gradient_descent_optimizer;

typedef struct momentum_optimizer {
    float learning_rate;
    float momentum_coefficient;
    void (* _Nullable minimize)(void * _Nonnull neural, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
} momentum_optimizer;

typedef struct ada_grad_optimizer {
    float learning_rate;
    float delta;
    dense * _Nullable dense;
    conv2d * _Nullable conv2d;
    void (* _Nullable minimize)(void * _Nonnull neural, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
} ada_grad_optimizer;

typedef struct rms_prop_optimizer {
    float learning_rate;
    float delta;
    float decay_rate;
    dense * _Nullable dense;
    conv2d * _Nullable conv2d;
    void (* _Nullable minimize)(void * _Nonnull neural, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
} rms_prop_optimizer;

typedef struct adam_optimizer {
    unsigned int time;
    float step_size;
    float delta;
    float decay_rate1;
    float decay_rate2;
    dense * _Nullable dense;
    conv2d * _Nullable conv2d;
    void (* _Nullable minimize)(void * _Nonnull neural, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
} adam_optimizer;

void gradient_descent_optimize(void * _Nonnull neural, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
void momentum_optimize(void * _Nonnull neural, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
void ada_grad_optimize(void * _Nonnull neural, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
void rms_prop_optimize(void * _Nonnull neural, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);
void adam_optimize(void * _Nonnull neural, tensor * _Nonnull  batch_features, tensor * _Nonnull batch_labels, unsigned int batch_size);

#endif /* Optimization_h */
