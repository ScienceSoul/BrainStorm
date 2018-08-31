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
    unsigned int max_number_nodes_in_layer;
    float eta, lambda;
    
    char supported_parameters[MAX_SUPPORTED_PARAMETERS][MAX_SHORT_STRING_LENGTH];
    int topology[MAX_NUMBER_NETWORK_LAYERS], split[2], classifications[MAX_NUMBER_NETWORK_LAYERS];
} dense_net_parameters;

typedef struct conv2d_net_parameters {
    unsigned int numberOfClassifications;
    unsigned int max_number_nodes_in_dense_layer;
    unsigned int max_propag_delta_entries;
    float eta, lambda;
    
    // For a 2D convolutional network, the first dimension of topology is the layer index,
    // the second dimension indicates in this order: the layer type (OUTPUT, CONVOLUTION, POOLING or OUTPUT),
    // the nunber of feature maps for this layer, the horiaontal dimension of the map, the vertical dimension
    // of the map, the horizontal dimension of the convolution kernel (or pooling), the vertical dimension of the
    // convolution kenel (or pooling), the horizontal stride, the vertical stride.
    // In case of a dense layer, the second dimension of topology has only one entry, i.e., the number of neurons in the layer
    int topology[MAX_NUMBER_NETWORK_LAYERS][8];
    
    int split[2], classifications[MAX_NUMBER_NETWORK_LAYERS];;
} conv2d_net_parameters;

#endif /* NetworkParams_h */
