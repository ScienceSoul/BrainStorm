//
//  NetworkConstructor.c
//  BrainStorm
//
//  Created by Hakime Seddik on 06/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include "NeuralNetwork.h"

void layer(void * _Nonnull neural, unsigned int nbNeurons, char * _Nonnull type, char * _Nullable activation) {
    
    bool firstTime = true;
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    if (firstTime) {
        nn->constructor->networkConstruction = true;
        firstTime = false;
    }
    
    if (nn->parameters->numberOfLayers >= MAX_NUMBER_NETWORK_LAYERS)
        fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow in network topology construction.");
    
    if (strcmp(type, "input") == 0) {
        if (activation != NULL) {
            fprintf(stdout, "%s: activation function passed to input layer. Will be ignored.", DEFAULT_CONSOLE_WRITER);
        }
        nn->parameters->topology[nn->parameters->numberOfLayers] = nbNeurons;
        nn->parameters->numberOfLayers++;
        
    } else if (strcmp(type, "hidden") == 0 || strcmp(type, "output") == 0) {
        nn->parameters->topology[nn->parameters->numberOfLayers] = nbNeurons;
        nn->parameters->numberOfLayers++;;
        
        if (strcmp(type, "hidden") == 0) {
            if (strcmp(activation, "softmax") == 0) fatal(DEFAULT_CONSOLE_WRITER, "the softmax function can only be used for the output units.");
        }
        
        if (strcmp(activation, "sigmoid") == 0) {
            nn->activationFunctions[nn->parameters->numberOfActivationFunctions] = sigmoid;
            nn->activationDerivatives[nn->parameters->numberOfActivationFunctions] = sigmoidPrime;
        } else if (strcmp(activation, "relu") == 0) {
            nn->activationFunctions[nn->parameters->numberOfActivationFunctions] = relu;
            nn->activationDerivatives[nn->parameters->numberOfActivationFunctions] = reluPrime;
        } else if (strcmp(activation, "tanh") == 0) {
            nn->activationFunctions[nn->parameters->numberOfActivationFunctions] = tan_h;
            nn->activationDerivatives[nn->parameters->numberOfActivationFunctions] = tanhPrime;
        } else if (strcmp(activation, "softmax") == 0) {
            nn->activationFunctions[nn->parameters->numberOfActivationFunctions] = softmax;
            nn->activationDerivatives[nn->parameters->numberOfActivationFunctions] = NULL;
        } else {
            fatal(DEFAULT_CONSOLE_WRITER, "unsupported or unrecognized activation function:", activation);
        }
        nn->parameters->numberOfActivationFunctions++;
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unknown layer type in network construction.");
    }
}

void split(void * _Nonnull neural, int n1, int n2) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    nn->parameters->split[0] = n1;
    nn->parameters->split[1] = n2;
}

void training_data(void * _Nonnull neural, char * _Nonnull str) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    unsigned int len = (unsigned int)strlen(str);
    if (len >= MAX_LONG_STRING_LENGTH) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when copying string in constructor");
    memcpy(nn->parameters->data, str, len*sizeof(char));
}

void classification(void * _Nonnull neural, int * _Nonnull vector, int n) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    if (n >= MAX_NUMBER_NETWORK_LAYERS) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when copying vector in constructor");
    memcpy(nn->parameters->classifications, vector, n*sizeof(int));
    nn->parameters->numberOfClassifications = n;
}

networkConstructor * _Nonnull allocateConstructor(void) {
    
    networkConstructor *constructor = (networkConstructor *)malloc(sizeof(networkConstructor));
    constructor->networkConstruction = false;
    constructor->layer = layer;
    constructor->split = split;
    constructor->training_data = training_data;
    constructor->classification = classification;
    return constructor;
}
