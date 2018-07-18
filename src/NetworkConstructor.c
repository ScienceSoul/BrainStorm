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

void set_layer(void * _Nonnull neural, unsigned int nbNeurons, char * _Nonnull type, char * _Nullable activation, regularizer_dict * _Nullable regularizer) {
    
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
        } else if (strcmp(activation, "leakyrelu") == 0) {
            strcpy(nn->parameters->activationFunctions[nn->parameters->numberOfActivationFunctions], "leakyrelu");
            nn->activationFunctions[nn->parameters->numberOfActivationFunctions] = leakyrelu;
            nn->activationDerivatives[nn->parameters->numberOfActivationFunctions] = leakyreluPrime;
        } else if (strcmp(activation, "elu") == 0) {
            strcpy(nn->parameters->activationFunctions[nn->parameters->numberOfActivationFunctions], "elu");
            nn->activationFunctions[nn->parameters->numberOfActivationFunctions] = elu;
            nn->activationDerivatives[nn->parameters->numberOfActivationFunctions] = eluPrime;
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
        
        // Add a regularizer if provided
        if (regularizer != NULL) {
            nn->parameters->lambda = regularizer->regularization_factor;
            nn->regularizer[nn->parameters->numberOfActivationFunctions] = regularizer->regularizer_func;
        } else nn->regularizer[nn->parameters->numberOfActivationFunctions] = nn->l0_regularizer;
        
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
}

void * _Nonnull set_optimizer(void * neural, optimizer_dict optimizer_dict) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    void * optimizer = NULL;
    
    if (strcmp(optimizer_dict.optimizer, "gradient descent") == 0) {
        nn->train->gradient_descent = (GradientDescentOptimizer *)malloc(sizeof(GradientDescentOptimizer));
        nn->train->gradient_descent->learning_rate = optimizer_dict.learning_rate;
        nn->parameters->eta = optimizer_dict.learning_rate;
        nn->train->gradient_descent->minimize = gradientDescentOptimizer;
        optimizer = (void *)nn->train->gradient_descent;
    } else if (strcmp(optimizer_dict.optimizer, "momentum") == 0) {
        nn->train->momentum = (MomentumOptimizer *)malloc(sizeof(MomentumOptimizer));
        nn->train->momentum->learning_rate = optimizer_dict.learning_rate;
        nn->train->momentum->momentum_coefficient = optimizer_dict.momentum;
        nn->parameters->eta = optimizer_dict.learning_rate;
        nn->train->momentum->minimize = momentumOptimizer;
        optimizer = (void *)nn->train->momentum;
    } else if (strcmp(optimizer_dict.optimizer, "adagrad") == 0) {
        nn->train->ada_grad = (AdaGradOptimizer *)malloc(sizeof(AdaGradOptimizer));
        nn->train->ada_grad->learning_rate = optimizer_dict.learning_rate;
        nn->train->ada_grad->delta = optimizer_dict.delta;
        nn->parameters->eta = optimizer_dict.learning_rate;;
        nn->train->ada_grad->costWeightDerivativeSquaredAccumulated = NULL;
        nn->train->ada_grad->costBiasDerivativeSquaredAccumulated = NULL;
        nn->train->ada_grad->minimize = adamOptimizer;
        optimizer = (void *)nn->train->ada_grad;
    } else if (strcmp(optimizer_dict.optimizer, "rmsprop") == 0) {
        nn->train->rms_prop = (RMSPropOptimizer *)malloc(sizeof(RMSPropOptimizer));
        nn->train->rms_prop->learning_rate = optimizer_dict.learning_rate;
        nn->train->rms_prop->decayRate = optimizer_dict.decay_rate1;
        nn->train->rms_prop->delta = optimizer_dict.delta;
        nn->parameters->eta = optimizer_dict.learning_rate;
        nn->train->rms_prop->costWeightDerivativeSquaredAccumulated = NULL;
        nn->train->rms_prop->costBiasDerivativeSquaredAccumulated = NULL;
        nn->train->rms_prop->minimize = rmsPropOptimizer;
        optimizer = (void *)nn->train->rms_prop;
    } else if (strcmp(optimizer_dict.optimizer, "adam") == 0) {
        nn->train->adam = (AdamOptimizer *)malloc(sizeof(AdamOptimizer));
        nn->train->adam->time = 0;
        nn->train->adam->stepSize = optimizer_dict.step_size;
        nn->train->adam->decayRate1 = optimizer_dict.decay_rate1;
        nn->train->adam->decayRate2 = optimizer_dict.decay_rate2;
        nn->train->adam->delta = optimizer_dict.delta;
        nn->parameters->eta = optimizer_dict.step_size;
        nn->train->adam->weightsBiasedFirstMomentEstimate = NULL;
        nn->train->adam->weightsBiasedSecondMomentEstimate = NULL;
        nn->train->adam->biasesBiasedFirstMomentEstimate = NULL;
        nn->train->adam->biasesBiasedSecondMomentEstimate = NULL;
        nn->train->adam->minimize = adamOptimizer;
        optimizer = (void *)nn->train->adam;
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unrecognized optimizer. Currently supported optimizers are: GradiendDescent, Momentum, AdaGrad, RMSProp and Adam.");
    }
    
    return optimizer;
}

networkConstructor * _Nonnull allocateConstructor(void) {
    
    networkConstructor *constructor = (networkConstructor *)malloc(sizeof(networkConstructor));
    constructor->networkConstruction = false;
    constructor->layer = set_layer;
    constructor->split = set_split;
    constructor->training_data = set_training_data;
    constructor->classification = set_classification;
    constructor->scalars = set_scalars;
    constructor->optimizer = set_optimizer;
    return constructor;
}
