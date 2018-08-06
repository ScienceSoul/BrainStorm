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

typedef struct dense_net_parameters {
    int epochs, miniBatchSize;
    unsigned int numberOfClassifications;
    unsigned int max_number_of_nodes_in_layer;
    float eta, lambda;
    
    char supported_parameters[MAX_SUPPORTED_PARAMETERS][MAX_SHORT_STRING_LENGTH];
    int topology[MAX_NUMBER_NETWORK_LAYERS], split[2], classifications[MAX_NUMBER_NETWORK_LAYERS];
} dense_net_parameters;

#endif /* NetworkParams_h */
