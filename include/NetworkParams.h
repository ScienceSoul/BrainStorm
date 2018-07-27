//
//  NetworkParams.h
//  BrainStorm
//
//  Created by Hakime Seddik on 06/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef NetworkParams_h
#define NetworkParams_h

#include "Utils.h"

typedef struct networkParameters {
    int epochs, miniBatchSize;
    unsigned int numberOfClassifications, numberOfActivationFunctions;
    unsigned int max_number_of_nodes_in_layer;
    unsigned int number_of_features;
    float eta, lambda;
    
    char supported_parameters[MAX_SUPPORTED_PARAMETERS][MAX_SHORT_STRING_LENGTH];
    char data[MAX_LONG_STRING_LENGTH], dataName[MAX_LONG_STRING_LENGTH];
    int topology[MAX_NUMBER_NETWORK_LAYERS], split[2], classifications[MAX_NUMBER_NETWORK_LAYERS];
    char activationFunctions[MAX_NUMBER_NETWORK_LAYERS][MAX_SHORT_STRING_LENGTH];
} networkParameters;

#endif /* NetworkParams_h */
