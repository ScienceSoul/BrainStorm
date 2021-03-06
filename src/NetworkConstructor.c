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

static void set_activation_function(activation_functions activation_functions_ref[MAX_NUMBER_NETWORK_LAYERS],
                    float (* _Nonnull activation_functions[MAX_NUMBER_NETWORK_LAYERS])(float z, float * _Nullable vec, unsigned int * _Nullable n),
                    float (* _Nonnull activation_derivatives[MAX_NUMBER_NETWORK_LAYERS])(float z),
                    layer_dict layer_dict, unsigned int idx) {
    
    if (layer_dict.activation == SIGMOID) {
        activation_functions_ref[idx] = SIGMOID;
        activation_functions[idx] = sigmoid;
        activation_derivatives[idx] = sigmoid_prime;
    } else if (layer_dict.activation == RELU) {
        activation_functions_ref[idx] = RELU;
        activation_functions[idx] = relu;
        activation_derivatives[idx] = relu_prime;
    } else if (layer_dict.activation == LEAKY_RELU) {
        activation_functions_ref[idx] = LEAKY_RELU;
        activation_functions[idx] = leakyrelu;
        activation_derivatives[idx] = leakyrelu_prime;
    } else if (layer_dict.activation == ELU) {
        activation_functions_ref[idx] = ELU;
        activation_functions[idx] = elu;
        activation_derivatives[idx] = elu_prime;
    } else if (layer_dict.activation == TANH) {
        activation_functions_ref[idx] = TANH;
        activation_functions[idx] = tan_h;
        activation_derivatives[idx] = tanh_prime;
    } else if (layer_dict.activation == SOFTPLUS) {
        activation_functions_ref[idx] = SOFTPLUS;
        activation_functions[idx] = softplus;
        activation_derivatives[idx] = softplus_prime;
    } else if (layer_dict.activation == SOFTMAX) {
        activation_functions_ref[idx] = SOFTMAX;
        activation_functions[idx] = softmax;
        activation_derivatives[idx] = NULL;
    } else if (layer_dict.activation == CUSTOM) {
        // Must be defined by the user
        activation_functions_ref[idx] = CUSTOM;
    } else {
        fprintf(stdout, "%s: activation function not given, default to sigmoid.\n", DEFAULT_CONSOLE_WRITER);
        activation_functions_ref[idx] = SIGMOID;
        activation_functions[idx] = sigmoid;
        activation_derivatives[idx] = sigmoid_prime;
    }
}

static void set_kernel_initializer(void (* _Nonnull kernel_initializers[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull object, float * _Nullable factor, char * _Nullable mode, bool * _Nullable uniform, int layer, int offset, float * _Nullable val),
                                   layer_dict layer_dict, unsigned int idx) {
    
    if (layer_dict.kernel_initializer == NULL) {
        fprintf(stdout, "%s: no initializer given to layer, default to standard nornmal distribution.\n", DEFAULT_CONSOLE_WRITER);
        kernel_initializers[idx] = xavier_initializer;
    } else {
        kernel_initializers[idx] = layer_dict.kernel_initializer;
    }
}

void set_feed(void * _Nonnull neural, layer_dict layer_dict) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    nn->constructor->network_construction = true;
    if (nn->network_num_layers != 0) {
        fatal(DEFAULT_CONSOLE_WRITER, "network topolgy error. The feeding layer must be created first.");
    }
    
    if (layer_dict.dimension == 1) {
        if (nn->dense == NULL) {
            fatal(DEFAULT_CONSOLE_WRITER, "feeding dimension equal to 1 must be used with a fully connected netwotk.");
        }
        nn->dense->parameters->topology[nn->network_num_layers] = layer_dict.shape;
        nn->num_channels = layer_dict.shape;
    } else if (layer_dict.dimension == 2 || layer_dict.dimension == 3) {
        if (layer_dict.dimension == 2) {
            unsigned int sh1;
            unsigned int sh2;
            if (layer_dict.shapes[0] != NULL) {
                sh1 = *(layer_dict.shapes[0]+0);
                sh2 = *(layer_dict.shapes[0]+1);
            } else {
                sh1 = layer_dict.shape;
                sh2 = layer_dict.shape;
            }
            
            nn->conv2d->parameters->topology[nn->network_num_layers][0] = FEED;
            nn->conv2d->parameters->topology[nn->network_num_layers][1] = layer_dict.filters;
            nn->conv2d->parameters->topology[nn->network_num_layers][2] = sh1;
            nn->conv2d->parameters->topology[nn->network_num_layers][3] = sh2;
            if (layer_dict.channels != NULL) {
                nn->num_channels = sh1 * sh2 * (*layer_dict.channels);
            } else {
                fatal(DEFAULT_CONSOLE_WRITER, "the nunber of channels must be provided for feeding dimensions higher than 2.");
            }
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
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    if (nn->network_num_layers >= MAX_NUMBER_NETWORK_LAYERS)
        fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow in network topology construction.");
    
    if (nn->is_dense_network) {
        nn->dense->parameters->topology[nn->network_num_layers] = layer_dict.num_neurons;
        
        // The activation function for this layer
        set_activation_function(nn->activation_functions_ref, nn->dense->activation_functions, nn->dense->activation_derivatives, layer_dict, nn->num_activation_functions);
        
        // The kernel initializer for this layer
        set_kernel_initializer(nn->kernel_initializers, layer_dict, nn->dense->num_dense_layers);
        nn->dense->num_dense_layers++;
      
    } else if (nn->is_conv2d_network) {
        nn->conv2d->parameters->topology[nn->network_num_layers][0] = FULLY_CONNECTED;
        nn->conv2d->parameters->topology[nn->network_num_layers][1] = layer_dict.num_neurons;
        
        set_activation_function(nn->activation_functions_ref, nn->conv2d->activation_functions, nn->conv2d->activation_derivatives, layer_dict, nn->num_activation_functions);
        
        set_kernel_initializer(nn->kernel_initializers, layer_dict, (nn->conv2d->num_conv2d_layers+nn->conv2d->num_dense_layers));
        nn->conv2d->num_dense_layers++;
        
        // Store the maximum number of nodes in the fully connected layers
        nn->conv2d->parameters->max_number_nodes_in_dense_layer =
                max((int)nn->conv2d->parameters->max_number_nodes_in_dense_layer, (int)layer_dict.num_neurons);
    }
    nn->network_num_layers++;
    
    // Add the regularizer if given
    if (regularizer != NULL) {
        if (nn->is_dense_network) {
            nn->dense->parameters->lambda = regularizer->regularization_factor;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->parameters->lambda = regularizer->regularization_factor;
        }
        nn->regularizer[nn->num_activation_functions] = regularizer->regularizer_func;
    } else nn->regularizer[nn->num_activation_functions] = nn->l0_regularizer;
    
    nn->num_activation_functions++;
    
    // If convolutional network, set this layer to fully connected inference operation
    // and to a fully connected backpropagation operation
    if (nn->is_conv2d_network) {
        nn->conv2d->inference_ops[nn->conv2d->num_infer_ops] = infer_fully_connected_op;
        nn->conv2d->num_infer_ops++;
        nn->conv2d->backpropag_ops[nn->conv2d->num_backpropag_ops] = backpropag_full_connected_op;
        nn->conv2d->num_backpropag_ops++;
    }
}

void set_layer_conv2d(void * _Nonnull neural, layer_dict layer_dict, regularizer_dict * _Nullable regulaizer) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    if (nn->network_num_layers >= MAX_NUMBER_NETWORK_LAYERS)
        fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow in network topology construction.");
    
    // Compute the size of each feature map using the kernel sizes, the strides and
    // whether padding is used
    if (layer_dict.padding == VALID) {

        // The dimensions of the layer on which convolution is applied
        // Typically given by previous layer
        int input_size_x = nn->conv2d->parameters->topology[nn->network_num_layers-1][2];
        int input_size_y = nn->conv2d->parameters->topology[nn->network_num_layers-1][3];
        
        // Horizontal and vertical size of the map after convolution
        unsigned int kh;
        unsigned int kw;
        unsigned int sh;
        unsigned int sw;
        if (layer_dict.kernel_sizes[0] != NULL && layer_dict.strides[0] != NULL) {
            kh = *(layer_dict.kernel_sizes[0]+0);
            kw = *(layer_dict.kernel_sizes[0]+1);
            sh = *(layer_dict.strides[0]+0);
            sw = *(layer_dict.strides[0]+1);
        } else if (layer_dict.kernel_sizes[0] != NULL && layer_dict.strides[0] == NULL) {
            kh = *(layer_dict.kernel_sizes[0]+0);
            kw = *(layer_dict.kernel_sizes[0]+1);
            sh = layer_dict.stride;
            sw = layer_dict.stride;
        } else if (layer_dict.kernel_sizes[0] == NULL && layer_dict.strides[0] != NULL) {
            kh = layer_dict.kernel_size;
            kw = layer_dict.kernel_size;
            sh = *(layer_dict.strides[0]+0);
            sw = *(layer_dict.strides[0]+1);
        } else {
            kh = layer_dict.kernel_size;
            kw = layer_dict.kernel_size;
            sh = layer_dict.stride;
            sw = layer_dict.stride;
        }
        int map_size_x = floorf((input_size_x-kh)/sh) + 1;
        int map_size_y = floorf((input_size_y-kw)/sw) + 1;
        
        nn->conv2d->parameters->topology[nn->network_num_layers][0] = CONVOLUTION;
        nn->conv2d->parameters->topology[nn->network_num_layers][1] = layer_dict.filters;
        nn->conv2d->parameters->topology[nn->network_num_layers][2] = map_size_x;
        nn->conv2d->parameters->topology[nn->network_num_layers][3] = map_size_y;
        nn->conv2d->parameters->topology[nn->network_num_layers][4] = kh;
        nn->conv2d->parameters->topology[nn->network_num_layers][5] = kw;
        nn->conv2d->parameters->topology[nn->network_num_layers][6] = sh;
        nn->conv2d->parameters->topology[nn->network_num_layers][7] = sw;
        nn->network_num_layers++;
        
    } else if (layer_dict.padding == SAME) {
        fatal(DEFAULT_CONSOLE_WRITER, "zero padding is not implemented yet.");
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unrecognized paddding option.");
    }
    
    // The kernel initializer for this layer
    set_kernel_initializer(nn->kernel_initializers, layer_dict, (nn->conv2d->num_conv2d_layers+nn->conv2d->num_dense_layers));
    nn->conv2d->num_conv2d_layers++;
    
    // The activation function for this layer
    set_activation_function(nn->activation_functions_ref, nn->conv2d->activation_functions, nn->conv2d->activation_derivatives, layer_dict, nn->num_activation_functions);
    
    // Add the regularizer if given
    if (regulaizer != NULL) {
        nn->conv2d->parameters->lambda = regulaizer->regularization_factor;
        nn->regularizer[nn->num_activation_functions] = regulaizer->regularizer_func;
    } else nn->regularizer[nn->num_activation_functions] = nn->l0_regularizer;
    
    nn->num_activation_functions++;
    
    // Set this layer to a convolutional inference operation and
    // to convolutional backpropagation operation
    nn->conv2d->inference_ops[nn->conv2d->num_infer_ops] = infer_convolution_op;
    nn->conv2d->num_infer_ops++;
    nn->conv2d->backpropag_ops[nn->conv2d->num_backpropag_ops] = backpropag_convolution_op;
    nn->conv2d->num_backpropag_ops++;
}

void set_layer_pool(void * _Nonnull neural, layer_dict layer_dict) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    if (nn->network_num_layers >= MAX_NUMBER_NETWORK_LAYERS)
    fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow in network topology construction.");
    
    if (layer_dict.padding == VALID) {
        // The dimensions of the layer on which pooling is applied
        // Typically given by previous layer
        int input_size_x = nn->conv2d->parameters->topology[nn->network_num_layers-1][2];
        int input_size_y = nn->conv2d->parameters->topology[nn->network_num_layers-1][3];
        
        // Horizontal and vertical size of the map after pooling
        unsigned int kh;
        unsigned int kw;
        unsigned int sh;
        unsigned int sw;
        if (layer_dict.kernel_sizes[0] != NULL && layer_dict.strides[0] != NULL) {
            kh = *(layer_dict.kernel_sizes[0]+0);
            kw = *(layer_dict.kernel_sizes[0]+1);
            sh = *(layer_dict.strides[0]+0);
            sw = *(layer_dict.strides[0]+1);
        } else if (layer_dict.kernel_sizes[0] != NULL && layer_dict.strides[0] == NULL) {
            kh = *(layer_dict.kernel_sizes[0]+0);
            kw = *(layer_dict.kernel_sizes[0]+1);
            sh = layer_dict.stride;
            sw = layer_dict.stride;
        } else if (layer_dict.kernel_sizes[0] == NULL && layer_dict.strides[0] != NULL) {
            kh = layer_dict.kernel_size;
            kw = layer_dict.kernel_size;
            sh = *(layer_dict.strides[0]+0);
            sw = *(layer_dict.strides[0]+1);
        } else {
            kh = layer_dict.kernel_size;
            kw = layer_dict.kernel_size;
            sh = layer_dict.stride;
            sw = layer_dict.stride;
        }
        int pool_size_x = floorf((input_size_x-kh)/sh) + 1;
        int pool_size_y = floorf((input_size_y-kw)/sw) + 1;
        
        nn->conv2d->parameters->topology[nn->network_num_layers][0] = POOLING;
        nn->conv2d->parameters->topology[nn->network_num_layers][1] = nn->conv2d->parameters->topology[nn->network_num_layers-1][1];
        nn->conv2d->parameters->topology[nn->network_num_layers][2] = pool_size_x;
        nn->conv2d->parameters->topology[nn->network_num_layers][3] = pool_size_y;
        nn->conv2d->parameters->topology[nn->network_num_layers][4] = kh;
        nn->conv2d->parameters->topology[nn->network_num_layers][5] = kw;
        nn->conv2d->parameters->topology[nn->network_num_layers][6] = sh;
        nn->conv2d->parameters->topology[nn->network_num_layers][7] = sw;
        
    } else if (layer_dict.padding == SAME) {
        fatal(DEFAULT_CONSOLE_WRITER, "zero padding is not implemented yet.");
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unrecognized paddding option.");
    }
    
    // Set this layer to a pooling inference operation and
    // to pooling backpropagation operation
    if (layer_dict.pooling_op == MAX_POOLING) {
        nn->conv2d->parameters->topology[nn->network_num_layers][8] = MAX_POOLING;
        nn->conv2d->inference_ops[nn->conv2d->num_infer_ops] = max_pooling_op;
        nn->conv2d->backpropag_ops[nn->conv2d->num_backpropag_ops] = backpropag_max_pooling_op;
    } else if (layer_dict.pooling_op == L2_POOLING) {
        nn->conv2d->parameters->topology[nn->network_num_layers][8] = L2_POOLING;
        nn->conv2d->inference_ops[nn->conv2d->num_infer_ops] = l2_pooling_op;
        nn->conv2d->backpropag_ops[nn->conv2d->num_backpropag_ops] = backpropag_l2_pooling_op;
    } else if (layer_dict.pooling_op == AVERAGE_POOLING) {
        nn->conv2d->parameters->topology[nn->network_num_layers][8] = AVERAGE_POOLING;
        nn->conv2d->inference_ops[nn->conv2d->num_infer_ops] = average_pooling_op;
        nn->conv2d->backpropag_ops[nn->conv2d->num_backpropag_ops] = backpropag_average_pooling_op;
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unrecognized pooling operation.");
    }
    nn->network_num_layers++;
    nn->conv2d->num_infer_ops++;
    nn->conv2d->num_backpropag_ops++;
    nn->conv2d->num_pooling_layers++;
}

void set_split(void * _Nonnull neural, int n1, int n2) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    if (nn->is_dense_network) {
        nn->dense->parameters->split[0] = n1;
        nn->dense->parameters->split[1] = n2;
    } else if (nn->is_conv2d_network) {
        nn->conv2d->parameters->split[0] = n1;
        nn->conv2d->parameters->split[1] = n2;
    }
}

void set_training_data(void * _Nonnull neural, char * _Nonnull str) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    unsigned int len = (unsigned int)strlen(str);
    if (len >= MAX_LONG_STRING_LENGTH) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when copying string in constructor");
    memcpy(nn->data_path, str, len*sizeof(char));
}

void set_classification(void * _Nonnull neural, int * _Nonnull vector, int n) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    if (n >= MAX_NUMBER_NETWORK_LAYERS) fatal(DEFAULT_CONSOLE_WRITER, "buffer overflow when copying vector in constructor");
    if (nn->is_dense_network) {
        memcpy(nn->dense->parameters->classifications, vector, n*sizeof(int));
        nn->dense->parameters->num_classifications = n;
    } else if (nn->is_conv2d_network) {
        memcpy(nn->conv2d->parameters->classifications, vector, n*sizeof(int));
        nn->conv2d->parameters->num_classifications = n;
    }
}


void set_scalars(void * _Nonnull neural, scalar_dict scalars) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    nn->dense->parameters->epochs = scalars.epochs;
    nn->dense->parameters->mini_batch_size = scalars.mini_batch_size;
}

void * _Nonnull set_optimizer(void * neural, optimizer_dict optimizer_dict) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    void * optimizer = NULL;
    
    if (optimizer_dict.optimizer == NULL) {
        fatal(DEFAULT_CONSOLE_WRITER, "Optimizer construction can't proceed with missing optimizer name.");
    }
    
    if (strcmp(optimizer_dict.optimizer, "gradient descent") == 0) {
        if (nn->is_dense_network) {
            nn->dense->train->gradient_descent =
                (gradient_descent_optimizer *)malloc(sizeof(gradient_descent_optimizer));
            nn->dense->train->gradient_descent->learning_rate = optimizer_dict.learning_rate;
            nn->dense->parameters->eta = optimizer_dict.learning_rate;
            nn->dense->train->gradient_descent->minimize = gradient_descent_optimize;
            optimizer = (void *)nn->dense->train->gradient_descent;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->train->gradient_descent =
                (gradient_descent_optimizer *)malloc(sizeof(gradient_descent_optimizer));
            nn->conv2d->train->gradient_descent->learning_rate = optimizer_dict.learning_rate;
            nn->conv2d->parameters->eta = optimizer_dict.learning_rate;
            nn->conv2d->train->gradient_descent->minimize = gradient_descent_optimize;
            optimizer = (void *)nn->conv2d->train->gradient_descent;
        }
    } else if (strcmp(optimizer_dict.optimizer, "momentum") == 0) {
        if (nn->is_dense_network) {
            nn->dense->train->momentum = (momentum_optimizer *)malloc(sizeof(momentum_optimizer));
            nn->dense->train->momentum->learning_rate = optimizer_dict.learning_rate;
            nn->dense->train->momentum->momentum_coefficient = optimizer_dict.momentum;
            nn->dense->parameters->eta = optimizer_dict.learning_rate;
            nn->dense->train->momentum->minimize = momentum_optimize;
            optimizer = (void *)nn->dense->train->momentum;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->train->momentum = (momentum_optimizer *)malloc(sizeof(momentum_optimizer));
            nn->conv2d->train->momentum->learning_rate = optimizer_dict.learning_rate;
            nn->conv2d->train->momentum->momentum_coefficient = optimizer_dict.momentum;
            nn->conv2d->parameters->eta = optimizer_dict.learning_rate;
            nn->conv2d->train->momentum->minimize = momentum_optimize;
            optimizer = (void *)nn->conv2d->train->momentum;
        }
    } else if (strcmp(optimizer_dict.optimizer, "adagrad") == 0) {
        if (nn->is_dense_network) {
            nn->dense->train->ada_grad = (ada_grad_optimizer *)malloc(sizeof(ada_grad_optimizer));
            nn->dense->train->ada_grad->learning_rate = optimizer_dict.learning_rate;
            nn->dense->train->ada_grad->delta = optimizer_dict.delta;
            nn->dense->parameters->eta = optimizer_dict.learning_rate;
            
            nn->dense->train->ada_grad->dense = (dense *)malloc(sizeof(dense));
            nn->dense->train->ada_grad->dense->cost_weight_derivative_squared_accumulated = NULL;
            nn->dense->train->ada_grad->dense->cost_bias_derivative_squared_accumulated = NULL;
            nn->dense->train->ada_grad->minimize = ada_grad_optimize;
            optimizer = (void *)nn->dense->train->ada_grad;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->train->ada_grad = (ada_grad_optimizer *)malloc(sizeof(ada_grad_optimizer));
            nn->conv2d->train->ada_grad->learning_rate = optimizer_dict.learning_rate;
            nn->conv2d->train->ada_grad->delta = optimizer_dict.delta;
            nn->conv2d->parameters->eta = optimizer_dict.learning_rate;
            
            nn->conv2d->train->ada_grad->conv2d = (conv2d *)malloc(sizeof(conv2d));
            nn->conv2d->train->ada_grad->dense = (dense *)malloc(sizeof(dense));
            nn->conv2d->train->ada_grad->conv2d->cost_weight_derivative_squared_accumulated = NULL;
            nn->conv2d->train->ada_grad->conv2d->cost_bias_derivative_squared_accumulated = NULL;
            nn->conv2d->train->ada_grad->dense->cost_weight_derivative_squared_accumulated = NULL;
            nn->conv2d->train->ada_grad->dense->cost_bias_derivative_squared_accumulated = NULL;
            nn->conv2d->train->ada_grad->minimize = ada_grad_optimize;
            optimizer = (void *)nn->conv2d->train->ada_grad;
        }
    } else if (strcmp(optimizer_dict.optimizer, "rmsprop") == 0) {
        if (nn->is_dense_network) {
            nn->dense->train->rms_prop = (rms_prop_optimizer *)malloc(sizeof(rms_prop_optimizer));
            nn->dense->train->rms_prop->learning_rate = optimizer_dict.learning_rate;
            nn->dense->train->rms_prop->decay_rate = optimizer_dict.decay_rate1;
            nn->dense->train->rms_prop->delta = optimizer_dict.delta;
            nn->dense->parameters->eta = optimizer_dict.learning_rate;
            
            nn->dense->train->rms_prop->dense = (dense *)malloc(sizeof(dense));
            nn->dense->train->rms_prop->dense->cost_weight_derivative_squared_accumulated = NULL;
            nn->dense->train->rms_prop->dense->cost_bias_derivative_squared_accumulated = NULL;
            nn->dense->train->rms_prop->minimize = rms_prop_optimize;
            optimizer = (void *)nn->dense->train->rms_prop;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->train->rms_prop = (rms_prop_optimizer *)malloc(sizeof(rms_prop_optimizer));
            nn->conv2d->train->rms_prop->learning_rate = optimizer_dict.learning_rate;
            nn->conv2d->train->rms_prop->decay_rate = optimizer_dict.decay_rate1;
            nn->conv2d->train->rms_prop->delta = optimizer_dict.delta;
            nn->conv2d->parameters->eta = optimizer_dict.learning_rate;
            
            nn->conv2d->train->rms_prop->conv2d = (conv2d *)malloc(sizeof(conv2d));
            nn->conv2d->train->rms_prop->dense = (dense *)malloc(sizeof(dense));
            nn->conv2d->train->rms_prop->conv2d->cost_weight_derivative_squared_accumulated = NULL;
            nn->conv2d->train->rms_prop->conv2d->cost_bias_derivative_squared_accumulated = NULL;
            nn->conv2d->train->rms_prop->dense->cost_weight_derivative_squared_accumulated = NULL;
            nn->conv2d->train->rms_prop->dense->cost_bias_derivative_squared_accumulated = NULL;
            nn->conv2d->train->rms_prop->minimize = rms_prop_optimize;
            optimizer = (void *)nn->conv2d->train->rms_prop;
        }
    } else if (strcmp(optimizer_dict.optimizer, "adam") == 0) {
        if (nn->is_dense_network) {
            nn->dense->train->adam = (adam_optimizer *)malloc(sizeof(adam_optimizer));
            nn->dense->train->adam->time = 0;
            nn->dense->train->adam->step_size = optimizer_dict.step_size;
            nn->dense->train->adam->decay_rate1 = optimizer_dict.decay_rate1;
            nn->dense->train->adam->decay_rate2 = optimizer_dict.decay_rate2;
            nn->dense->train->adam->delta = optimizer_dict.delta;
            nn->dense->parameters->eta = optimizer_dict.step_size;
            
            nn->dense->train->adam->dense = (dense *)malloc(sizeof(dense));
            nn->dense->train->adam->dense->weights_biased_first_moment_estimate = NULL;
            nn->dense->train->adam->dense->weights_biased_second_moment_estimate = NULL;
            nn->dense->train->adam->dense->biases_biased_first_moment_estimate = NULL;
            nn->dense->train->adam->dense->biases_biased_second_moment_estimate = NULL;
            nn->dense->train->adam->minimize = adam_optimize;
            optimizer = (void *)nn->dense->train->adam;
        } else if (nn->is_conv2d_network) {
            nn->conv2d->train->adam = (adam_optimizer *)malloc(sizeof(adam_optimizer));
            nn->conv2d->train->adam->time = 0;
            nn->conv2d->train->adam->step_size = optimizer_dict.step_size;
            nn->conv2d->train->adam->decay_rate1 = optimizer_dict.decay_rate1;
            nn->conv2d->train->adam->decay_rate2 = optimizer_dict.decay_rate2;
            nn->conv2d->train->adam->delta = optimizer_dict.delta;
            nn->conv2d->parameters->eta = optimizer_dict.step_size;
            
            nn->conv2d->train->adam->conv2d = (conv2d *)malloc(sizeof(conv2d));
            nn->conv2d->train->adam->dense = (dense *)malloc(sizeof(dense));
            nn->conv2d->train->adam->conv2d->weights_biased_first_moment_estimate = NULL;
            nn->conv2d->train->adam->conv2d->weights_biased_second_moment_estimate = NULL;
            nn->conv2d->train->adam->conv2d->biases_biased_first_moment_estimate = NULL;
            nn->conv2d->train->adam->conv2d->biases_biased_second_moment_estimate = NULL;
            nn->conv2d->train->adam->dense->weights_biased_first_moment_estimate = NULL;
            nn->conv2d->train->adam->dense->weights_biased_second_moment_estimate = NULL;
            nn->conv2d->train->adam->dense->biases_biased_first_moment_estimate = NULL;
            nn->conv2d->train->adam->dense->biases_biased_second_moment_estimate = NULL;
            nn->conv2d->train->adam->minimize = adam_optimize;
            optimizer = (void *)nn->conv2d->train->adam;
        }
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unrecognized optimizer. Currently supported optimizers are: GradiendDescent, Momentum, AdaGrad, RMSProp and Adam.");
    }
    
    return optimizer;
}

network_constructor * _Nonnull allocate_constructor(void) {
    
    network_constructor *constructor = (network_constructor *)malloc(sizeof(network_constructor));
    constructor->network_construction = false;
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
