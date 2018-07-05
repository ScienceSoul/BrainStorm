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

void * _Nonnull allocateActivationNode(void);
void * _Nonnull allocateAffineTransformationNode(void);
void * _Nonnull allocateCostWeightDerivativeNode(void);
void * _Nonnull allocateCostBiaseDerivativeNode(void);

float * _Nonnull initMatrices(int * _Nonnull topology, unsigned int numberOfLayers, bool gaussian);
float * _Nonnull initVectors(int * _Nonnull topology, unsigned int numberOfLayers, bool gaussian);

void * _Nonnull initNetworkActivations(int * _Nonnull ntLayers, unsigned int numberOfLayers);
void * _Nonnull initNetworkAffineTransformations(int * _Nonnull ntLayers, unsigned int numberOfLayers);

void * _Nonnull initNetworkCostWeightDerivatives(int * _Nonnull ntLayers, unsigned int numberOfLayers);
void * _Nonnull initNetworkCostBiaseDerivatives(int * _Nonnull ntLayers, unsigned int numberOfLayers);

int loadParameters(void * _Nonnull self, const char * _Nonnull paraFile);

#endif /* NetworkUtils_h */
