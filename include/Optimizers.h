//
//  Optimization.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 28/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef Optimization_h
#define Optimization_h

typedef struct GradientDescentOptimizer {
    float learning_rate;
    void (* _Nullable minimize)(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size);
} GradientDescentOptimizer;

typedef struct MomentumOptimizer {
    float learning_rate;
    float momentum_coefficient;
    void (* _Nullable minimize)(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size);
} MomentumOptimizer;

typedef struct AdaGradOptimizer {
    float learning_rate;
    float delta;
    float * _Nullable costWeightDerivativeSquaredAccumulated;
    float * _Nullable costBiasDerivativeSquaredAccumulated;
    void (* _Nullable minimize)(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size);
} AdaGradOptimizer;

typedef struct RMSPropOptimizer {
    float learning_rate;
    float delta;
    float decayRate;
    float * _Nullable costWeightDerivativeSquaredAccumulated;
    float * _Nullable costBiasDerivativeSquaredAccumulated;
    void (* _Nullable minimize)(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size);
} RMSPropOptimizer;

typedef struct AdamOptimizer {
    unsigned int time;
    float stepSize;
    float delta;
    float decayRate1;
    float decayRate2;
    float * _Nullable weightsBiasedFirstMomentEstimate;
    float * _Nullable weightsBiasedSecondMomentEstimate;
    float * _Nullable biasesBiasedFirstMomentEstimate;
    float * _Nullable biasesBiasedSecondMomentEstimate;
    void (* _Nullable minimize)(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size);
} AdamOptimizer;

void gradientDescentOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size);
void momentumOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size);
void adaGradOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size);
void rmsPropOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size);
void adamOptimizer(void * _Nonnull neural, float * _Nonnull * _Nonnull  mini_batch, unsigned int batch_size);

#endif /* Optimization_h */