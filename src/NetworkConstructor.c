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
#include "Conv2DNetOps.h"

static void set_activation_function(activation_functions activationFunctionsRef[MAX_NUMBER_NETWORK_LAYERS],
                    float (* _Nonnull activationFunctions[MAX_NUMBER_NETWORK_LAYERS])(float z, float * _Nullable vec, unsigned int * _Nullable n),
                    float (* _Nonnull activationDerivatives[MAX_NUMBER_NETWORK_LAYERS])(float z),
                    layer_dict layer_dict, unsigned int idx) {
    
    if (layer_dict.activation == SIGMOID) {
        activationFunctionsRef[idx] = SIGMOID;
        activationFunctions[idx] = sigmoid;
        activationDerivatives[idx] = sigmoidPrime;
    } else if (layer_dict.activation == RELU) {
        activationFunctionsRef[idx] = RELU;
        activationFunctions[idx] = relu;
        activationDerivatives[idx] = reluPrime;
    } else if (layer_dict.activation == LEAKY_RELU) {
        activationFunctionsRef[idx] = LEAKY_RELU;
        activationFunctions[idx] = leakyrelu;
        activationDerivatives[idx] = leakyreluPrime;
    } else if (layer_dict.activation == ELU) {
        activationFunctionsRef[idx] = ELU;
        activationFunctions[idx] = elu;
        activationDerivatives[idx] = eluPrime;
    } else if (layer_dict.activation == TANH) {
        activationFunctionsRef[idx] = TANH;
        activationFunctions[idx] = tan_h;
        activationDerivatives[idx] = tanhPrime;
    } else if (layer_dict.activation == SOFTMAX) {
        activationFunctionsRef[idx] = SOFTMAX;
        activationFunctions[idx] = softmax;
        activationDerivatives[idx] = NULL;
    } else {
        fprintf(stdout, "%s: activation function not given, default to sigmoid.\n", DEFAULT_CONSOLE_WRITER);
        activationFunctionsRef[idx] = SIGMOID;
        activationFunctions[idx] = sigmoid;
        activationDerivatives[idx] = sigmoidPrime;
    }
}

static void set_kernel_initializer(void (* _Nonnull kernelInitializers[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, void * _Nonnull kernel,  int l, int offset),
                                   layer_dict layer_dict, unsigned int idx) {
    
    if (layer_dict.kernel_initializer == NULL) {
        fprintf(stdout, "%s: no initializer given to layer, default to standard nornmal distribution.\n", DEFAULT_CONSOLE_WRITER);
        kernelInitializers[idx] = standard_normal_initializer;
    } else {
        kernelInitializers[idx] = layer_dict.kernel_initializer;
    }
}

void set_feed(void * _Nonnull neural, unsigned int shape[_Nonnull 3], unsigned int dimension, unsigned int * _Nullable num_channels) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    nn->constructor->networkConstruction = true;
    if (nn->network_num_layers != 0) {
        fatal(DEFAULT_CONSOLE_WRITER, "network topolgy error. The feeding layer must be created first.");
    }
    
    if (dimension == 1) {
        if (nn->dense == NULL) {
            fatal(DEFAULT_CONSOLE_WRITER, "feeding dimension equal to 1 must be used with a fully connected netwotk.");
        }
        nn->dense->parameters->topology[nn->network_num_layers] = shape[0];
        nn->num_channels = shape[0];
    } else if (dimension == 2 || dimension == 3) {
        if (num_channels == NULL) {
            fatal(DEFAULT_CONSOLE_WRITER, "the nunber of channels must be provided for feeding dimension higher than 2.");
        }
        if (dimension == 2) {
            if (nn->conv2d == NULL) {
                fatal(DEFAULT_CONSOLE_WRITER, "feeding dimension >=2 must be used with a convolutional netwotk.");
            }
            nn->conv2d->parameters->topology[nn->network_num_layers][0] = FEED;
            nn->conv2d->parameters->topology[nn->network_num_layers][1] = 1;
            nn->conv2d->parameters->topology[nn->network_num_layers][2] = shape[0];
            nn->conv2d->parameters->topology[nn->network_num_layers][3] = shape[1];
            nn->num_channels = shape[0] * shape[1] * (*num_channels);
        } else {
            //TODO: needs implementation
            fatal(DEFAULT_CONSOLE_WRITER, "3D convolution is not supported yet.");
            //nn->num_channels = shape[0] * shape[1] * shape[2] * (*num_channels);
        }
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "imput dimension in network feeding not supported. Must be 1, 2 or 3.");
    }
    nn->network_num_layers++;
}

void set_layer_dense(void * _Nonnull neural, layer_dict layer_dict, regularizer_dict * _Nullable regularizer) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    if (nn->network_num_layers >= MAX_NUMBER_NETWORK_LAYERS)
        fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow in network topology construction.");
    
    if (nn->is_dense_network) {
        nn->dense->parameters->topology[nn->network_num_layers] = layer_dict.num_neurons;
        
        // The activation function for this layer
        set_activation_function(nn->activationFunctionsRef, nn->dense->activationFunctions, nn->dense->activationDerivatives, layer_dict, nn->num_activation_functions);
        
        // The kernel initializer for this layer
        set_kernel_initializer(nn->dense->kernelInitializers, layer_dict, nn->dense->num_dense_layers);
        nn->dense->num_dense_layers++;
      
    } else if (nn->is_conv2d_network) {
        nn->conv2d->parameters->topology[nn->network_num_layers][0] = FULLY_CONNECTED;
        nn->conv2d->parameters->topology[nn->network_num_layers][1] = layer_dict.num_neurons;
        
        set_activation_function(nn->activationFunctionsRef, nn->conv2d->activationFunctions, nn->conv2d->activationDerivatives, layer_dict, nn->num_activation_functions);
        
        set_kernel_initializer(nn->conv2d->kernelInitializers, layer_dict, (nn->conv2d->num_conv2d_layers+nn->conv2d->num_dense_layers));
        nn->conv2d->num_dense_layers++;
    }
    nn->network_num_layers++;
    
    // Add the regularizer if given
    if (regularizer != NULL) {
        nn->dense->parameters->lambda = regularizer->regularization_factor;
        nn->regularizer[nn->num_activation_functions] = regularizer->regularizer_func;
    } else nn->regularizer[nn->num_activation_functions] = nn->l0_regularizer;
    
    nn->num_activation_functions++;
    
    // Set this layer to fully connected operation if convolutional network
    if (nn->is_conv2d_network) {
        nn->conv2d->layersOps[nn->conv2d->num_ops] = full_connected_ops;
        nn->conv2d->num_ops++;
    }
}

void set_layer_conv2d(void * _Nonnull neural, layer_dict layer_dict, regularizer_dict * _Nullable regulaizer) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    if (nn->network_num_layers >= MAX_NUMBER_NETWORK_LAYERS)
        fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow in network topology construction.");
    
    // Compute the size of each feature map using the kernel sizes, the strides and
    // whether padding is used
    if (layer_dict.padding == NO_PADDING) {
        int pos = 0;
        int map_size_x = 1; // First neuron in map layer (horizontal)
        int map_size_y = 1; // First neuron in map layer (vertical)

        // The input dimensions are defined by the previous layer
        int input_size_x = nn->conv2d->parameters->topology[nn->network_num_layers-1][2];
        int input_size_y = nn->conv2d->parameters->topology[nn->network_num_layers-1][3];
        
        // Horizontal
        while (1) {
            pos += layer_dict.strides[0];
            if ((pos+layer_dict.kernel_size[0]) > input_size_x) {
                break;
            } else map_size_x++;
        }
        
        // Vertical
        pos = 0;
        while (1) {
            pos += layer_dict.strides[1];
            if ((pos+layer_dict.kernel_size[1]) > input_size_y) {
                break;
            } else map_size_y++;
        }
        
        nn->conv2d->parameters->topology[nn->network_num_layers][0] = CONVOLUTION;
        nn->conv2d->parameters->topology[nn->network_num_layers][1] = layer_dict.filters;
        nn->conv2d->parameters->topology[nn->network_num_layers][2] = map_size_x;
        nn->conv2d->parameters->topology[nn->network_num_layers][3] = map_size_y;
        nn->conv2d->parameters->topology[nn->network_num_layers][4] = layer_dict.kernel_size[0];
        nn->conv2d->parameters->topology[nn->network_num_layers][5] = layer_dict.kernel_size[1];
        nn->conv2d->parameters->topology[nn->network_num_layers][6] = layer_dict.strides[0];
        nn->conv2d->parameters->topology[nn->network_num_layers][7] = layer_dict.strides[1];
        nn->network_num_layers++;
        
    } else if (layer_dict.padding == ZERO_PADDING) {
        fatal(DEFAULT_CONSOLE_WRITER, "zero padding is not implemented yet.");
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unrecognized paddding option.");
    }
    
    // The kernel initializer for this layer
    set_kernel_initializer(nn->conv2d->kernelInitializers, layer_dict, (nn->conv2d->num_conv2d_layers+nn->conv2d->num_dense_layers));
    nn->conv2d->num_conv2d_layers++;
    
    // The activation function for this layer
    set_activation_function(nn->activationFunctionsRef, nn->conv2d->activationFunctions, nn->conv2d->activationDerivatives, layer_dict, nn->num_activation_functions);
    
    // Add the regularizer if given
    if (regulaizer != NULL) {
        nn->conv2d->parameters->lambda = regulaizer->regularization_factor;
        nn->regularizer[nn->num_activation_functions] = regulaizer->regularizer_func;
    } else nn->regularizer[nn->num_activation_functions] = nn->l0_regularizer;
    
    nn->num_activation_functions++;
    
    // Set this layer to a convolution operation
    nn->conv2d->layersOps[nn->conv2d->num_ops] = convolution_ops;
    nn->conv2d->num_ops++;
}

void set_layer_pool(void * _Nonnull neural, layer_dict layer_dict) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    if (nn->network_num_layers >= MAX_NUMBER_NETWORK_LAYERS)
    fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow in network topology construction.");
    
    if (layer_dict.padding == NO_PADDING) {
        int pos = 0;
        int pool_size_x = 1; // First neuron in pool layer (horizontal)
        int pool_size_y = 1; // First neuron in pool layer (vertical)
        
        // The input dimensions are defined by the previous layer
        int input_size_x = nn->conv2d->parameters->topology[nn->network_num_layers-1][2];
        int input_size_y = nn->conv2d->parameters->topology[nn->network_num_layers-1][3];
        
        // Horizontal
        while (1) {
            pos = pos + (layer_dict.strides[0]+layer_dict.kernel_size[0]-1);
            if ((pos+layer_dict.kernel_size[0]) > input_size_x) {
                break;
            } else pool_size_x++;
        }
        
        // Vertical
        pos = 0;
        while (1) {
            pos = pos + (layer_dict.strides[1]+layer_dict.kernel_size[1]-1);
            if ((pos+layer_dict.kernel_size[1]) > input_size_y) {
                break;
            } else pool_size_y++;
        }
        
        nn->conv2d->parameters->topology[nn->network_num_layers][0] = POOLING;
        nn->conv2d->parameters->topology[nn->network_num_layers][1] = nn->conv2d->parameters->topology[nn->network_num_layers-1][1];
        nn->conv2d->parameters->topology[nn->network_num_layers][2] = pool_size_x;
        nn->conv2d->parameters->topology[nn->network_num_layers][3] = pool_size_y;
        nn->conv2d->parameters->topology[nn->network_num_layers][4] = layer_dict.kernel_size[0];
        nn->conv2d->parameters->topology[nn->network_num_layers][5] = layer_dict.kernel_size[1];
        nn->conv2d->parameters->topology[nn->network_num_layers][6] = layer_dict.strides[0];
        nn->conv2d->parameters->topology[nn->network_num_layers][7] = layer_dict.strides[1];
        nn->network_num_layers++;
        
    } else if (layer_dict.padding == ZERO_PADDING) {
        fatal(DEFAULT_CONSOLE_WRITER, "zero padding is not implemented yet.");
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unrecognized paddding option.");
    }
    
    // Set this layer to a pooling operation
    if (layer_dict.pooling_op == MAX_POOLING) {
        nn->conv2d->layersOps[nn->conv2d->num_ops] = max_pool;
    } else if (layer_dict.pooling_op == L2_POOLING) {
        nn->conv2d->layersOps[nn->conv2d->num_ops] = l2_pool;
    } else if (layer_dict.pooling_op == AVERAGE_POOLING) {
        nn->conv2d->layersOps[nn->conv2d->num_ops] = average_pool;
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unrecognized pooling operation.");
    }
    nn->conv2d->num_ops++;
    nn->conv2d->num_pooling_layers++;
}

void set_split(void * _Nonnull neural, int n1, int n2) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    if (nn->is_dense_network) {
        nn->dense->parameters->split[0] = n1;
        nn->dense->parameters->split[1] = n2;
    } else if (nn->is_conv2d_network) {
        nn->conv2d->parameters->split[0] = n1;
        nn->conv2d->parameters->split[1] = n2;
    }
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
    if (nn->is_dense_network) {
        memcpy(nn->dense->parameters->classifications, vector, n*sizeof(int));
        nn->dense->parameters->numberOfClassifications = n;
    } else if (nn->is_conv2d_network) {
        memcpy(nn->conv2d->parameters->classifications, vector, n*sizeof(int));
        nn->conv2d->parameters->numberOfClassifications = n;
    }
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
        if (nn->is_dense_network) {
            nn->dense->train->gradient_descent =
                (GradientDescentOptimizer *)malloc(sizeof(GradientDescentOptimizer));
            nn->dense->train->gradient_descent->learning_rate = optimizer_dict.learning_rate;
            nn->dense->parameters->eta = optimizer_dict.learning_rate;
            nn->dense->train->gradient_descent->minimize = gradientDescentOptimizer;
            optimizer = (void *)nn->dense->train->gradient_descent;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->train->gradient_descent =
                (GradientDescentOptimizer *)malloc(sizeof(GradientDescentOptimizer));
            nn->conv2d->train->gradient_descent->learning_rate = optimizer_dict.learning_rate;
            nn->conv2d->parameters->eta = optimizer_dict.learning_rate;
            nn->conv2d->train->gradient_descent->minimize = gradientDescentOptimizer;
            optimizer = (void *)nn->conv2d->train->gradient_descent;
        }
    } else if (strcmp(optimizer_dict.optimizer, "momentum") == 0) {
        if (nn->is_dense_network) {
            nn->dense->train->momentum = (MomentumOptimizer *)malloc(sizeof(MomentumOptimizer));
            nn->dense->train->momentum->learning_rate = optimizer_dict.learning_rate;
            nn->dense->train->momentum->momentum_coefficient = optimizer_dict.momentum;
            nn->dense->parameters->eta = optimizer_dict.learning_rate;
            nn->dense->train->momentum->minimize = momentumOptimizer;
            optimizer = (void *)nn->dense->train->momentum;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->train->momentum = (MomentumOptimizer *)malloc(sizeof(MomentumOptimizer));
            nn->conv2d->train->momentum->learning_rate = optimizer_dict.learning_rate;
            nn->conv2d->train->momentum->momentum_coefficient = optimizer_dict.momentum;
            nn->conv2d->parameters->eta = optimizer_dict.learning_rate;
            nn->conv2d->train->momentum->minimize = momentumOptimizer;
            optimizer = (void *)nn->conv2d->train->momentum;
        }
    } else if (strcmp(optimizer_dict.optimizer, "adagrad") == 0) {
        if (nn->is_dense_network) {
            nn->dense->train->ada_grad = (AdaGradOptimizer *)malloc(sizeof(AdaGradOptimizer));
            nn->dense->train->ada_grad->learning_rate = optimizer_dict.learning_rate;
            nn->dense->train->ada_grad->delta = optimizer_dict.delta;
            nn->dense->parameters->eta = optimizer_dict.learning_rate;;
            nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated = NULL;
            nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated = NULL;
            nn->dense->train->ada_grad->minimize = adaGradOptimizer;
            optimizer = (void *)nn->dense->train->ada_grad;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->train->ada_grad = (AdaGradOptimizer *)malloc(sizeof(AdaGradOptimizer));
            nn->conv2d->train->ada_grad->learning_rate = optimizer_dict.learning_rate;
            nn->conv2d->train->ada_grad->delta = optimizer_dict.delta;
            nn->conv2d->parameters->eta = optimizer_dict.learning_rate;
            nn->conv2d->train->ada_grad->conv2d->costWeightDerivativeSquaredAccumulated = NULL;
            nn->conv2d->train->ada_grad->conv2d->costBiasDerivativeSquaredAccumulated = NULL;
            nn->conv2d->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated = NULL;
            nn->conv2d->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated = NULL;
            nn->conv2d->train->ada_grad->minimize = adaGradOptimizer;
            optimizer = (void *)nn->conv2d->train->ada_grad;
        }
    } else if (strcmp(optimizer_dict.optimizer, "rmsprop") == 0) {
        if (nn->is_dense_network) {
            nn->dense->train->rms_prop = (RMSPropOptimizer *)malloc(sizeof(RMSPropOptimizer));
            nn->dense->train->rms_prop->learning_rate = optimizer_dict.learning_rate;
            nn->dense->train->rms_prop->decayRate = optimizer_dict.decay_rate1;
            nn->dense->train->rms_prop->delta = optimizer_dict.delta;
            nn->dense->parameters->eta = optimizer_dict.learning_rate;
            nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated = NULL;
            nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated = NULL;
            nn->dense->train->rms_prop->minimize = rmsPropOptimizer;
            optimizer = (void *)nn->dense->train->rms_prop;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->train->rms_prop = (RMSPropOptimizer *)malloc(sizeof(RMSPropOptimizer));
            nn->conv2d->train->rms_prop->learning_rate = optimizer_dict.learning_rate;
            nn->conv2d->train->rms_prop->decayRate = optimizer_dict.decay_rate1;
            nn->conv2d->train->rms_prop->delta = optimizer_dict.delta;
            nn->conv2d->parameters->eta = optimizer_dict.learning_rate;
            nn->conv2d->train->rms_prop->conv2d->costWeightDerivativeSquaredAccumulated = NULL;
            nn->conv2d->train->rms_prop->conv2d->costBiasDerivativeSquaredAccumulated = NULL;
            nn->conv2d->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated = NULL;
            nn->conv2d->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated = NULL;
            nn->conv2d->train->rms_prop->minimize = rmsPropOptimizer;
            optimizer = (void *)nn->conv2d->train->rms_prop;
        }
    } else if (strcmp(optimizer_dict.optimizer, "adam") == 0) {
        if (nn->is_dense_network) {
            nn->dense->train->adam = (AdamOptimizer *)malloc(sizeof(AdamOptimizer));
            nn->dense->train->adam->time = 0;
            nn->dense->train->adam->stepSize = optimizer_dict.step_size;
            nn->dense->train->adam->decayRate1 = optimizer_dict.decay_rate1;
            nn->dense->train->adam->decayRate2 = optimizer_dict.decay_rate2;
            nn->dense->train->adam->delta = optimizer_dict.delta;
            nn->dense->parameters->eta = optimizer_dict.step_size;
            nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate = NULL;
            nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate = NULL;
            nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate = NULL;
            nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate = NULL;
            nn->dense->train->adam->minimize = adamOptimizer;
            optimizer = (void *)nn->dense->train->adam;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->train->adam = (AdamOptimizer *)malloc(sizeof(AdamOptimizer));
            nn->conv2d->train->adam->time = 0;
            nn->conv2d->train->adam->stepSize = optimizer_dict.step_size;
            nn->conv2d->train->adam->decayRate1 = optimizer_dict.decay_rate1;
            nn->conv2d->train->adam->decayRate2 = optimizer_dict.decay_rate2;
            nn->conv2d->train->adam->delta = optimizer_dict.delta;
            nn->conv2d->parameters->eta = optimizer_dict.step_size;
            nn->conv2d->train->adam->conv2d->weightsBiasedFirstMomentEstimate = NULL;
            nn->conv2d->train->adam->conv2d->weightsBiasedSecondMomentEstimate = NULL;
            nn->conv2d->train->adam->conv2d->biasesBiasedFirstMomentEstimate = NULL;
            nn->conv2d->train->adam->conv2d->biasesBiasedSecondMomentEstimate = NULL;
            nn->conv2d->train->adam->dense->weightsBiasedFirstMomentEstimate = NULL;
            nn->conv2d->train->adam->dense->weightsBiasedSecondMomentEstimate = NULL;
            nn->conv2d->train->adam->dense->biasesBiasedFirstMomentEstimate = NULL;
            nn->conv2d->train->adam->dense->biasesBiasedSecondMomentEstimate = NULL;
            nn->conv2d->train->adam->minimize = adamOptimizer;
            optimizer = (void *)nn->conv2d->train->adam;
        }
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
    constructor->layer_conv2d = set_layer_conv2d;
    constructor->layer_pool = set_layer_pool;
    
    constructor->split = set_split;
    constructor->training_data = set_training_data;
    constructor->classification = set_classification;
    
    constructor->scalars = set_scalars;
    constructor->optimizer = set_optimizer;
    return constructor;
}
