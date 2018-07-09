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

void set_layer(void * _Nonnull neural, unsigned int nbNeurons, char * _Nonnull type, char * _Nullable activation) {
    
    static bool firstTime = true;
    
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
        if (activation == NULL) fatal(DEFAULT_CONSOLE_WRITER, "activation function is null in constructor.");
        
        nn->parameters->topology[nn->parameters->numberOfLayers] = nbNeurons;
        nn->parameters->numberOfLayers++;;
        
        if (strcmp(type, "hidden") == 0) {
            if (strcmp(activation, "softmax") == 0) fatal(DEFAULT_CONSOLE_WRITER, "the softmax function can only be used for the output units.");
        }
        
        if (strcmp(activation, "sigmoid") == 0) {
            strcpy(nn->parameters->activationFunctions[nn->parameters->numberOfActivationFunctions], "sigmoid");
            nn->activationFunctions[nn->parameters->numberOfActivationFunctions] = sigmoid;
            nn->activationDerivatives[nn->parameters->numberOfActivationFunctions] = sigmoidPrime;
        } else if (strcmp(activation, "relu") == 0) {
            strcpy(nn->parameters->activationFunctions[nn->parameters->numberOfActivationFunctions], "relu");
            nn->activationFunctions[nn->parameters->numberOfActivationFunctions] = relu;
            nn->activationDerivatives[nn->parameters->numberOfActivationFunctions] = reluPrime;
        } else if (strcmp(activation, "tanh") == 0) {
            strcpy(nn->parameters->activationFunctions[nn->parameters->numberOfActivationFunctions], "tanh");
            nn->activationFunctions[nn->parameters->numberOfActivationFunctions] = tan_h;
            nn->activationDerivatives[nn->parameters->numberOfActivationFunctions] = tanhPrime;
        } else if (strcmp(activation, "softmax") == 0) {
            strcpy(nn->parameters->activationFunctions[nn->parameters->numberOfActivationFunctions], "softmax");
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

void set_split(void * _Nonnull neural, int n1, int n2) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    nn->parameters->split[0] = n1;
    nn->parameters->split[1] = n2;
}

void set_training_data(void * _Nonnull neural, char * _Nonnull str) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    unsigned int len = (unsigned int)strlen(str);
    if (len >= MAX_LONG_STRING_LENGTH) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when copying string in constructor");
    memcpy(nn->parameters->data, str, len*sizeof(char));
}

void set_classification(void * _Nonnull neural, int * _Nonnull vector, int n) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    if (n >= MAX_NUMBER_NETWORK_LAYERS) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when copying vector in constructor");
    memcpy(nn->parameters->classifications, vector, n*sizeof(int));
    nn->parameters->numberOfClassifications = n;
}

void set_scalars(void * _Nonnull neural, scalar_dict scalars) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    nn->parameters->epochs = scalars.epochs;
    nn->parameters->miniBatchSize = scalars.mini_batch_size;
    nn->parameters->eta = scalars.learning_rate;
    nn->parameters->lambda = scalars.regularization_factor;
    nn->parameters->mu = scalars.momentum_coefficient;
}

void set_adaptive_learning(void * _Nonnull neural, char * _Nonnull method, adaptive_dict adaptive_dict) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    if (strcmp(method, "adagrad") == 0) {
        nn->adaGrad = (AdaGrad *)malloc(sizeof(AdaGrad));
        nn->adaGrad->delta = adaptive_dict.scalar;
        nn->adaGrad->costWeightDerivativeSquaredAccumulated = NULL;
        nn->adaGrad->costBiasDerivativeSquaredAccumulated = NULL;
        nn->adapativeLearningRateMethod = ADAGRAD;
    } else if (strcmp(method, "rmsprop") == 0) {
         nn->rmsProp = (RMSProp *)malloc(sizeof(RMSProp));
        nn->rmsProp->decayRate = adaptive_dict.vector[0];
        nn->rmsProp->delta = adaptive_dict.vector[1];
        nn->rmsProp->costWeightDerivativeSquaredAccumulated = NULL;
        nn->rmsProp->costBiasDerivativeSquaredAccumulated = NULL;
        nn->adapativeLearningRateMethod = RMSPROP;
    } else if (strcmp(method, "adam") == 0) {
        nn->adam = (Adam *)malloc(sizeof(Adam));
        nn->adam->time = 0;
        nn->adam->stepSize = adaptive_dict.vector[0];
        nn->adam->decayRate1 = adaptive_dict.vector[1];
        nn->adam->decayRate2 = adaptive_dict.vector[2];
        nn->adam->delta = adaptive_dict.vector[3];
        nn->adam->weightsBiasedFirstMomentEstimate = NULL;
        nn->adam->weightsBiasedSecondMomentEstimate = NULL;
        nn->adam->biasesBiasedFirstMomentEstimate = NULL;
        nn->adam->biasesBiasedSecondMomentEstimate = NULL;
        nn->adapativeLearningRateMethod = ADAM;
    }
}

networkConstructor * _Nonnull allocateConstructor(void) {
    
    networkConstructor *constructor = (networkConstructor *)malloc(sizeof(networkConstructor));
    constructor->networkConstruction = false;
    constructor->layer = set_layer;
    constructor->split = set_split;
    constructor->training_data = set_training_data;
    constructor->classification = set_classification;
    constructor->scalars = set_scalars;
    constructor->adaptive_learning = set_adaptive_learning;
    return constructor;
}
