//
//  Optimization.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 28/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef Optimization_h
#define Optimization_h

enum {
    ADAGRAD = 1,
    RMSPROP,
    ADAM
};

typedef struct AdaGrad {
    float delta;
    float * _Nullable costWeightDerivativeSquaredAccumulated;
    float * _Nullable costBiasDerivativeSquaredAccumulated;
} AdaGrad;

typedef struct RMSProp {
    float delta;
    float decayRate;
    float * _Nullable costWeightDerivativeSquaredAccumulated;
    float * _Nullable costBiasDerivativeSquaredAccumulated;
} RMSProp;

typedef struct Adam {
    unsigned int time;
    float stepSize;
    float delta;
    float decayRate1;
    float decayRate2;
    float * _Nullable weightsBiasedFirstMomentEstimate;
    float * _Nullable weightsBiasedSecondMomentEstimate;
    float * _Nullable biasesBiasedFirstMomentEstimate;
    float * _Nullable biasesBiasedSecondMomentEstimate;
} Adam;

#endif /* Optimization_h */
