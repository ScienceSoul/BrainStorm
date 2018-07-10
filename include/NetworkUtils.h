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

typedef struct tensor_dict {
    unsigned int rank;
    bool init;
    char * _Nullable init_stategy;
} tensor_dict;

void * _Nonnull allocateActivationNode(void);
void * _Nonnull allocateAffineTransformationNode(void);
void * _Nonnull allocateCostWeightDerivativeNode(void);
void * _Nonnull allocateCostBiaseDerivativeNode(void);

float * _Nonnull tensor(void * _Nonnull self, tensor_dict tensor_dict);
float * _Nonnull initMatrices(void * _Nonnull self, bool init, char * _Nonnull strategy);
float * _Nonnull initVectors(void * _Nonnull self, bool init, char * _Nonnull strategy);

void * _Nonnull initNetworkActivations(int * _Nonnull ntLayers, unsigned int numberOfLayers);
void * _Nonnull initNetworkAffineTransformations(int * _Nonnull ntLayers, unsigned int numberOfLayers);

void * _Nonnull initNetworkCostWeightDerivatives(int * _Nonnull ntLayers, unsigned int numberOfLayers);
void * _Nonnull initNetworkCostBiaseDerivatives(int * _Nonnull ntLayers, unsigned int numberOfLayers);

int loadParametersFromImputFile(void * _Nonnull self, const char * _Nonnull paraFile);

#endif /* NetworkUtils_h */
