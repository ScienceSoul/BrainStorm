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

void set_feed(void * _Nonnull neural, unsigned int nbNeurons) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    nn->constructor->networkConstruction = true;
    if (nn->network_num_layers != 0) {
        fatal(DEFAULT_CONSOLE_WRITER, "network topolgy error. The feeding layer must be created first.");
    }
    nn->dense->parameters->topology[nn->network_num_layers] = nbNeurons;
    nn->network_num_layers++;
    
    nn->num_channels = nbNeurons;
}

void set_layer_dense(void * _Nonnull neural, unsigned int nbNeurons, layer_dict layer_dict, regularizer_dict * _Nullable regularizer) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    if (nn->network_num_layers >= MAX_NUMBER_NETWORK_LAYERS)
        fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow in network topology construction.");
    
    if (layer_dict.activation == NULL) fatal(DEFAULT_CONSOLE_WRITER, "activation function is null in constructor.");
    
    nn->dense->parameters->topology[nn->network_num_layers] = nbNeurons;
    nn->network_num_layers++;;
    
    if (strcmp(layer_dict.activation, "sigmoid") == 0) {
        strcpy(nn->activationFunctionsStr[nn->num_activation_functions], "sigmoid");
        nn->dense->activationFunctions[nn->num_activation_functions] = sigmoid;
        nn->dense->activationDerivatives[nn->num_activation_functions] = sigmoidPrime;
    } else if (strcmp(layer_dict.activation, "relu") == 0) {
        strcpy(nn->activationFunctionsStr[nn->num_activation_functions], "relu");
        nn->dense->activationFunctions[nn->num_activation_functions] = relu;
        nn->dense->activationDerivatives[nn->num_activation_functions] = reluPrime;
    } else if (strcmp(layer_dict.activation, "leakyrelu") == 0) {
        strcpy(nn->activationFunctionsStr[nn->num_activation_functions], "leakyrelu");
        nn->dense->activationFunctions[nn->num_activation_functions] = leakyrelu;
        nn->dense->activationDerivatives[nn->num_activation_functions] = leakyreluPrime;
    } else if (strcmp(layer_dict.activation, "elu") == 0) {
        strcpy(nn->activationFunctionsStr[nn->num_activation_functions], "elu");
        nn->dense->activationFunctions[nn->num_activation_functions] = elu;
        nn->dense->activationDerivatives[nn->num_activation_functions] = eluPrime;
    } else if (strcmp(layer_dict.activation, "tanh") == 0) {
        strcpy(nn->activationFunctionsStr[nn->num_activation_functions], "tanh");
        nn->dense->activationFunctions[nn->num_activation_functions] = tan_h;
        nn->dense->activationDerivatives[nn->num_activation_functions] = tanhPrime;
    } else if (strcmp(layer_dict.activation, "softmax") == 0) {
        strcpy(nn->activationFunctionsStr[nn->num_activation_functions], "softmax");
        nn->dense->activationFunctions[nn->num_activation_functions] = softmax;
        nn->dense->activationDerivatives[nn->num_activation_functions] = NULL;
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unsupported or unrecognized activation function:", layer_dict.activation);
    }
    
    if (layer_dict.kernel_initializer == NULL) {
        fprintf(stdout, "%s: no initializer given to layer, default to standard nornmal distribution.\n", DEFAULT_CONSOLE_WRITER);
        nn->dense->kernelInitializers[nn->dense->num_dense_layers] = standard_normal_initializer;
    } else {
        nn->dense->kernelInitializers[nn->dense->num_dense_layers] = layer_dict.kernel_initializer;
    }
    nn->dense->num_dense_layers++;
    
    // Add the regularizer if given
    if (regularizer != NULL) {
        nn->dense->parameters->lambda = regularizer->regularization_factor;
        nn->regularizer[nn->num_activation_functions] = regularizer->regularizer_func;
    } else nn->regularizer[nn->num_activation_functions] = nn->l0_regularizer;
    
    nn->num_activation_functions++;
}

void set_split(void * _Nonnull neural, int n1, int n2) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    nn->dense->parameters->split[0] = n1;
    nn->dense->parameters->split[1] = n2;
}

void set_training_data(void * _Nonnull neural, char * _Nonnull str) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    unsigned int len = (unsigned int)strlen(str);
    if (len >= MAX_LONG_STRING_LENGTH) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when copying string in constructor");
    memcpy(nn->dataPath, str, len*sizeof(char));
}

void set_classification(void * _Nonnull neural, int * _Nonnull vector, int n) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    if (n >= MAX_NUMBER_NETWORK_LAYERS) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when copying vector in constructor");
    memcpy(nn->dense->parameters->classifications, vector, n*sizeof(int));
    nn->dense->parameters->numberOfClassifications = n;
}

void set_scalars(void * _Nonnull neural, scalar_dict scalars) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    nn->dense->parameters->epochs = scalars.epochs;
    nn->dense->parameters->miniBatchSize = scalars.mini_batch_size;
}

void * _Nonnull set_optimizer(void * neural, optimizer_dict optimizer_dict) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    void * optimizer = NULL;
    
    if (strcmp(optimizer_dict.optimizer, "gradient descent") == 0) {
        nn->dense->train->gradient_descent = (GradientDescentOptimizer *)malloc(sizeof(GradientDescentOptimizer));
        nn->dense->train->gradient_descent->learning_rate = optimizer_dict.learning_rate;
        nn->dense->parameters->eta = optimizer_dict.learning_rate;
        nn->dense->train->gradient_descent->minimize = gradientDescentOptimizer;
        optimizer = (void *)nn->dense->train->gradient_descent;
    } else if (strcmp(optimizer_dict.optimizer, "momentum") == 0) {
        nn->dense->train->momentum = (MomentumOptimizer *)malloc(sizeof(MomentumOptimizer));
        nn->dense->train->momentum->learning_rate = optimizer_dict.learning_rate;
        nn->dense->train->momentum->momentum_coefficient = optimizer_dict.momentum;
        nn->dense->parameters->eta = optimizer_dict.learning_rate;
        nn->dense->train->momentum->minimize = momentumOptimizer;
        optimizer = (void *)nn->dense->train->momentum;
    } else if (strcmp(optimizer_dict.optimizer, "adagrad") == 0) {
        nn->dense->train->ada_grad = (AdaGradOptimizer *)malloc(sizeof(AdaGradOptimizer));
        nn->dense->train->ada_grad->learning_rate = optimizer_dict.learning_rate;
        nn->dense->train->ada_grad->delta = optimizer_dict.delta;
        nn->dense->parameters->eta = optimizer_dict.learning_rate;;
        nn->dense->train->ada_grad->costWeightDerivativeSquaredAccumulated = NULL;
        nn->dense->train->ada_grad->costBiasDerivativeSquaredAccumulated = NULL;
        nn->dense->train->ada_grad->minimize = adamOptimizer;
        optimizer = (void *)nn->dense->train->ada_grad;
    } else if (strcmp(optimizer_dict.optimizer, "rmsprop") == 0) {
        nn->dense->train->rms_prop = (RMSPropOptimizer *)malloc(sizeof(RMSPropOptimizer));
        nn->dense->train->rms_prop->learning_rate = optimizer_dict.learning_rate;
        nn->dense->train->rms_prop->decayRate = optimizer_dict.decay_rate1;
        nn->dense->train->rms_prop->delta = optimizer_dict.delta;
        nn->dense->parameters->eta = optimizer_dict.learning_rate;
        nn->dense->train->rms_prop->costWeightDerivativeSquaredAccumulated = NULL;
        nn->dense->train->rms_prop->costBiasDerivativeSquaredAccumulated = NULL;
        nn->dense->train->rms_prop->minimize = rmsPropOptimizer;
        optimizer = (void *)nn->dense->train->rms_prop;
    } else if (strcmp(optimizer_dict.optimizer, "adam") == 0) {
        nn->dense->train->adam = (AdamOptimizer *)malloc(sizeof(AdamOptimizer));
        nn->dense->train->adam->time = 0;
        nn->dense->train->adam->stepSize = optimizer_dict.step_size;
        nn->dense->train->adam->decayRate1 = optimizer_dict.decay_rate1;
        nn->dense->train->adam->decayRate2 = optimizer_dict.decay_rate2;
        nn->dense->train->adam->delta = optimizer_dict.delta;
        nn->dense->parameters->eta = optimizer_dict.step_size;
        nn->dense->train->adam->weightsBiasedFirstMomentEstimate = NULL;
        nn->dense->train->adam->weightsBiasedSecondMomentEstimate = NULL;
        nn->dense->train->adam->biasesBiasedFirstMomentEstimate = NULL;
        nn->dense->train->adam->biasesBiasedSecondMomentEstimate = NULL;
        nn->dense->train->adam->minimize = adamOptimizer;
        optimizer = (void *)nn->dense->train->adam;
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unrecognized optimizer. Currently supported optimizers are: GradiendDescent, Momentum, AdaGrad, RMSProp and Adam.");
    }
    
    return optimizer;
}

networkConstructor * _Nonnull allocateConstructor(void) {
    
    networkConstructor *constructor = (networkConstructor *)malloc(sizeof(networkConstructor));
    constructor->networkConstruction = false;
    constructor->feed = set_feed;
    constructor->layer_dense = set_layer_dense;
    constructor->split = set_split;
    constructor->training_data = set_training_data;
    constructor->classification = set_classification;
    constructor->scalars = set_scalars;
    constructor->optimizer = set_optimizer;
    return constructor;
}
