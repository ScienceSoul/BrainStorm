//
//  NetworkUtils.c
//  FeedforwardNT
//
//  Created by Hakime Seddik on 26/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#include <stdio.h>
#include "NetworkUtils.h"
#include "NeuralNetwork.h"
#include "Memory.h"
#include "Parsing.h"

typedef float (* get_value_func_ptr)(float *val);

static float random_normal(float *dummy) {
    return randn(0.0f, 1.0f);
}

static float variance_scaling_uniform(float *r_sigma) {
    return random_uniform(-(*r_sigma), *r_sigma);
}

static float variance_scaling_normal(float *r_sigma) {
    return randn(0.0f, *r_sigma);
}

static float get_func(float *val) {
    return *val;
}

void set_value(void * _Nonnull object, int layer, int offset, float * _Nullable val, get_value_func_ptr func_ptr) {
    
    tensor *tensor_object = (tensor *)object;
    
    float *value = NULL;
    if (func_ptr == get_func || func_ptr == variance_scaling_uniform || func_ptr == variance_scaling_normal) {
        value = val;
    }
    
    if (tensor_object->rank == 0) {
        tensor_object->val[offset] = func_ptr(value);
    } else if(tensor_object->rank == 1) {
        int n = tensor_object->shape[layer][0][0];
        for (int i=0; i<n; i++) {
            tensor_object->val[offset+i] = func_ptr(value);
        }
    } else if (tensor_object->rank == 2) {
        int m = tensor_object->shape[layer][0][0];
        int n = tensor_object->shape[layer][0][0];
        for (int i = 0; i<m; i++) {
            for (int j=0; j<n; j++) {
                tensor_object->val[offset+((i*n)+j)] = func_ptr(value);
            }
        }
    } else if (tensor_object->rank == 4) {
        int m = tensor_object->shape[layer][2][0];
        int n = tensor_object->shape[layer][3][0];
        
        int stride1 = 0;
        for (int l=0; l<tensor_object->shape[layer][0][0]; l++) {
            int stride2 = 0;
            for (int ll=0; ll<tensor_object->shape[layer][1][0]; ll++) {
                for (int i = 0; i<m; i++) {
                    for (int j=0; j<n; j++) {
                        tensor_object->val[offset+(stride1+(stride2+((i*n)+j)))] = func_ptr(value);
                    }
                }
                stride2 = stride2 + (m * n);
            }
            stride1 = stride1 + (tensor_object->shape[layer][1][0] * m * n);
        }
    } else if (tensor_object->rank == 5) {
        int m = tensor_object->shape[layer][2][0];
        int n = tensor_object->shape[layer][3][0];
        int q = tensor_object->shape[layer][4][0];
        
        int stride1 = 0;
        for (int l=0; l<tensor_object->shape[l][0][0]; l++) {
            int stride2 = 0;
            for (int ll=0; ll<tensor_object->shape[layer][1][0]; ll++) {
                for (int i = 0; i<m; i++) {
                    for (int j=0; j<n; j++) {
                        for(int k=0; k<q; k++) {
                            tensor_object->val[offset+(stride1+(stride2+((i*n*q)+(j*q)+k)))] = func_ptr(value);
                        }
                    }
                }
                stride2 = stride2 + (m * n * q);
            }
            stride1 = stride1 + (tensor_object->shape[layer][1][0] * m * n * q);
        }
    }
}

void variance_scaling_initializer(void * _Nonnull object, float * _Nullable factor, char * _Nullable mode, bool * _Nullable uniform, int layer, int offset, float * _Nullable val) {
    
    tensor *tensor_object = (tensor *)object;
    
    if (mode == NULL) fatal(DEFAULT_CONSOLE_WRITER, " initializer mode is NULL");
    
    float FACTOR = 2.0f;
    if (factor != NULL) FACTOR = *factor;
    
    float fan_in = 0.0f;
    float fan_out = 0.0f;
    if (tensor_object->rank > 0) {
        if (tensor_object->rank > 1) {
            if (tensor_object->rank == 2) {
                fan_in = (float)tensor_object->shape[layer][1][0];
                fan_out = (float)tensor_object->shape[layer][0][0];
            } else {
                fan_in = (float)tensor_object->shape[layer][0][0];
                fan_out = (float)tensor_object->shape[layer][1][0];
            }
        } else {
            fan_in = (float)tensor_object->shape[layer][0][0];
            fan_out = (float)tensor_object->shape[layer][0][0];
        }
        
        if (tensor_object->rank > 2) {
            for (int i=2; i<tensor_object->rank; i++) {
                fan_in *= (float)tensor_object->shape[layer][i][0];
                fan_out *= (float)tensor_object->shape[layer][i][0];
            }
        }
        
    } else {
        fan_in = 1.0f;
        fan_out = 1.0f;
    }
    
    float n = 0.0f;
    if (strcmp(mode, "FAN_IN") == 0) {
        // Count only number of input connections
        n = fan_in;
    } else if (strcmp(mode, "FAN_OUT") == 0) {
        // Count only number of output connections
        n = fan_out;
    } else if (strcmp(mode, "FAN_AVG") == 0) {
        // Average number of inputs and output connections.
        n = (fan_out + fan_out) / 2.0f;
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "unrecognized mode in initializer");
    }
    
    get_value_func_ptr func_prt = NULL;
    float r_sigma = 0.0f;
    if (*uniform) {
        r_sigma = sqrtf(3.0f * FACTOR / n);
        func_prt = variance_scaling_uniform;
    } else {
        r_sigma = sqrtf(FACTOR / n);
        func_prt = variance_scaling_normal;
    }
    
    set_value((void *)tensor_object, layer, offset, &r_sigma, func_prt);
}

void xavier_initializer(void * _Nonnull object, float * _Nullable factor, char * _Nullable mode, bool * _Nullable uniform, int layer, int offset, float * _Nullable val) {
    
    float fac = 1.0f;
    bool uni_distribution = true;
    variance_scaling_initializer(object, &fac, "FAN_AVG", &uni_distribution, layer, offset, NULL);
}

void random_normal_initializer(void * _Nonnull object, float * _Nullable factor, char * _Nullable mode, bool * _Nullable uniform, int layer, int offset, float * _Nullable val) {
    
    tensor *tensor_object = (tensor *)object;
    set_value((void *)tensor_object, layer, offset, NULL, &random_normal);
}

void value_initializer(void * _Nonnull object, float * _Nullable factor, char * _Nullable mode, bool * _Nullable uniform, int layer, int offset, float * _Nullable val) {
    
    tensor *tensor_object = (tensor *)object;
    set_value((void *)tensor_object, layer, offset, val, &get_func);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// Tensor allocation //////////////////////////////////////////
//      This routine allocates a 1D, 2D, 3D, 4D or 5D tensor with the possibility to linearize several
//      tensors on a single data allocation (usefull for allocating accross network layers).
//      Xavier-He initialization is only possible for 2D and 4D tensors.
//      The tensor shape, rank and the flattening length are given inside the tensor_dict structure.
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void * _Nullable tensor_create(void * _Nullable self, tensor_dict tensor_dict) {
    
    brain_storm_net *nn = NULL;
    if (self != NULL) {
        nn = (brain_storm_net *)self;
    }
    
    if (tensor_dict.rank > MAX_TENSOR_RANK) {
        fatal(DEFAULT_CONSOLE_WRITER, "tensors with rank > 5 are not supported.");
    }
    
    tensor *tensor_object = (tensor *)malloc(sizeof(tensor));
    if (tensor_object == NULL) {
        return NULL;
    }
    memset(*tensor_object->shape, 1, (MAX_NUMBER_NETWORK_LAYERS*MAX_TENSOR_RANK*1));
    
    int tensor_length = 0;
    for (int l=0; l<tensor_dict.flattening_length; l++) {
        int dim = 1;
        for (int i=0; i<tensor_dict.rank; i++) {
            dim = dim * tensor_dict.shape[l][i][0];
        }
        tensor_length = tensor_length + dim;
    }
    
    if (tensor_length == 0) {
        fatal(DEFAULT_CONSOLE_WRITER, "tensor allocation error. Tensor size is zero.");
    }
    
    tensor_object->val = (float *)malloc(tensor_length*sizeof(float));
    if (tensor_object->val == NULL) {
        return NULL;
    }
    memcpy(**tensor_object->shape, **tensor_dict.shape, (MAX_NUMBER_NETWORK_LAYERS*MAX_TENSOR_RANK*1)*sizeof(int));
    tensor_object->rank = tensor_dict.rank;
    fprintf(stdout, "%s: tensor allocation: allocate %f (MB)\n", DEFAULT_CONSOLE_WRITER, ((float)tensor_length*sizeof(float))/(float)(1024*1024));
    
    if (tensor_dict.init_weights) {
        int offset = 0;
        for (int l=0; l<tensor_dict.flattening_length; l++) {
            // One single tensor step
            int step = 1;
            for (int i=0; i<tensor_object->rank; i++) {
                step = step * tensor_dict.shape[l][i][0];
            }
            
            if (nn->kernel_initializers[l] == xavier_initializer) {
                nn->kernel_initializers[l]((void *)tensor_object, NULL, NULL, NULL, l, offset, NULL);
            } else if (nn->kernel_initializers[l] == variance_scaling_initializer) {
                bool uniform = false;
                nn->kernel_initializers[l]((void *)tensor_object, NULL, "FAN_IN", &uniform, l, offset, NULL);
            } else if (nn->kernel_initializers[l] == random_normal_initializer) {
                nn->kernel_initializers[l]((void *)tensor_object, NULL, NULL, NULL, l, offset, NULL);
            }
            
            offset = offset + step;
        }
    } else if (tensor_dict.init_with_value) {
        int offset = 0;
        for (int l=0; l<tensor_dict.flattening_length; l++) {
            int step = 1;
             for (int i=0; i<tensor_object->rank; i++) {
                 step = step * tensor_dict.shape[l][i][0];
             }
            
            value_initializer((void *)tensor_object, NULL, NULL, NULL, l, offset, &tensor_dict.init_value);
            offset = offset + step;
        }
    } else {
        memset(tensor_object->val, 0.0f, tensor_length*sizeof(float));
    }
    
    return (void *)tensor_object;
}

tensor_dict * _Nonnull init_tensor_dict(void) {
    
    tensor_dict *dict = (tensor_dict *)malloc(sizeof(tensor_dict));
    
    *(dict) = (tensor_dict){.init_weights=false, .init_with_value=false, .full_connected=false, .flattening_length=1, .rank =0, .init_value=0.0f};
    memset(**dict->shape, 0, (MAX_NUMBER_NETWORK_LAYERS*MAX_TENSOR_RANK)*sizeof(int));
    return dict;
}

// ----------------------------------------------
// Special function to shuffle inputs and labels
// ----------------------------------------------
void __attribute__((overloadable))shuffle(void * _Nonnull features, void * _Nullable labels, int num_classifications, int * _Nullable num_features) {
    
    static bool firstTime = true;
    
    tensor *input1 = (tensor *)features;
    tensor *input2 = (tensor *)labels;
    
    static int dim1 = 1;
    static int dim2 = 0;
    static int num_inputs = 0;
    if (firstTime) {
        
        if (input1->rank == 4) {
            for (int i=1; i<input1->rank; i++) {
                dim1 = dim1 * input1->shape[0][i][0];
            }
            num_inputs = input1->shape[0][0][0];
        } else if (input1->rank == 1) {
            if (num_features == NULL) fatal(DEFAULT_CONSOLE_WRITER, "missing argument for the number of inputs in shuffle routine.");
            dim1 = *num_features;
            num_inputs = input1->shape[0][0][0] / *num_features;
        } else fatal(DEFAULT_CONSOLE_WRITER, "shuffle routine only for 1D and 4D tensors.");
        
        if (num_classifications > 0) {
            dim2 = num_classifications;
        } else dim2 = 1;
        
        firstTime = false;
    }
    
    float t1[dim1];
    float t2[dim2];
    
    if (num_inputs> 1) {
        int stride1 = 0;
        int stride2 = 0;
        for (int i=0; i<num_inputs-1; i++) {
            int j = i + rand() / (RAND_MAX / (num_inputs - i) + 1);
            int jump1 = max((j-1),0) * dim1;
            int jump2 = max((j-1),0) * dim2;
            for (int k=0; k<dim1; k++) {
                t1[k] = input1->val[jump1+k];
            }
            for (int k=0; k<dim2; k++) {
                t2[k] = input2->val[jump2+k];
            }
            
            for (int k=0; k<dim1; k++) {
                input1->val[jump1+k] = input1->val[stride1+k];
            }
            for (int k=0; k<dim2; k++) {
                input2->val[jump2+k] = input2->val[stride2+k];
            }
            
            for (int k=0; k<dim1; k++) {
                input1->val[stride1+k] = t1[k];
            }
            for (int k=0; k<dim2; k++) {
                input2->val[stride2+k] = t2[k];
            }
            
            stride1 = stride1 + dim1;
            stride2 = stride2 + dim2;
        }
    }
}


int load_parameters_from_imput_file(void * _Nonnull self, const char * _Nonnull para_file) {
    
    definition *definitions = NULL;
    
    fprintf(stdout, "%s: load the network and its input parameters...\n", DEFAULT_CONSOLE_WRITER);
    
    definitions = get_definitions(self, para_file, "define");
    if (definitions == NULL) {
        fatal(DEFAULT_CONSOLE_WRITER, "problem finding any parameter definition.");
    }
    
    brain_storm_net *nn = (brain_storm_net *)self;
    
    short FOUND_DATA           = 0;
    short FOUND_TOPOLOGY       = 0;
    short FOUND_ACTIVATIONS    = 0;
    short FOUND_SPLIT          = 0;
    short FOUND_CLASSIFICATION = 0;
    short FOUND_REGULARIZATION = 0;
    short FOUND_OPTIMIZER      = 0;
    
    definition *pt = definitions;
    while (pt != NULL) {
        dictionary *field = pt->field;
        while (field != NULL) {
            bool found = false;
            for (int i=0; i<MAX_SUPPORTED_PARAMETERS; i++) {
                if (strcmp(field->key, nn->dense->parameters->supported_parameters[i]) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) fatal(DEFAULT_CONSOLE_WRITER, "key for parameter not recognized:", field->key);
            
            if (strcmp(field->key, "data_name") == 0) {
                strcpy(nn->data_name, field->value);
                
            } else if (strcmp(field->key, "data") == 0) {
                strcpy(nn->data_path, field->value);
                FOUND_DATA = 1;
                
            } else if (strcmp(field->key, "topology") == 0) {
                unsigned int len = MAX_NUMBER_NETWORK_LAYERS;
                parse_argument(field->value, field->key, nn->dense->parameters->topology, &nn->network_num_layers, &len);
                FOUND_TOPOLOGY = 1;
                
            } else if (strcmp(field->key, "activations") == 0) {
                
                // Activation functions:
                //      sigmoid ->  Logistic sigmoid
                //      relu    ->  Rectified linear unit
                //      tanh    ->  Hyperbolic tangent
                //      softmax ->  Softmax unit
                
                if (FOUND_TOPOLOGY == 0) {
                    fatal(DEFAULT_CONSOLE_WRITER, "incorrect parameters definition order, the topology is not defined yet. ");
                }
                
                char activationFunctionsStr[MAX_NUMBER_NETWORK_LAYERS][MAX_SHORT_STRING_LENGTH];
                unsigned int len = MAX_NUMBER_NETWORK_LAYERS;
                parse_argument(field->value, field->key, activationFunctionsStr, &nn->num_activation_functions, &len);
                
                if (nn->num_activation_functions > 1 && nn->num_activation_functions < nn->network_num_layers-1) {
                    fatal(DEFAULT_CONSOLE_WRITER, "the number of activation functions in parameters is too low. Can't resolve how to use the provided activations. ");
                }
                if (nn->num_activation_functions > nn->network_num_layers-1) {
                    fprintf(stdout, "%s: too many activation functions given to network. Will ignore the extra ones.\n", DEFAULT_CONSOLE_WRITER);
                }
                
                if (nn->num_activation_functions == 1) {
                    if (strcmp(activationFunctionsStr[0], "softmax") == 0) {
                        fatal(DEFAULT_CONSOLE_WRITER, "the softmax function can only be used for the output units, not for the entire network.");
                    }
                    for (int i=0; i<nn->network_num_layers-1; i++) {
                        if (strcmp(activationFunctionsStr[0], "sigmoid") == 0) {
                            nn->activation_functions_ref[i] = SIGMOID;
                            nn->dense->activation_functions[i] = sigmoid;
                            nn->dense->activation_derivatives[i] = sigmoid_prime;
                        } else if (strcmp(activationFunctionsStr[0], "relu") == 0) {
                            nn->activation_functions_ref[i] = RELU;
                            nn->dense->activation_functions[i] = relu;
                            nn->dense->activation_derivatives[i] = relu_prime;
                        } else if (strcmp(activationFunctionsStr[0], "leakyrelu") == 0) {
                            nn->activation_functions_ref[i] = LEAKY_RELU;
                            nn->dense->activation_functions[i] = leakyrelu;
                            nn->dense->activation_derivatives[i] = leakyrelu_prime;
                        } else if (strcmp(activationFunctionsStr[0], "elu") == 0) {
                            nn->activation_functions_ref[i] = ELU;
                            nn->dense->activation_functions[i] = elu;
                            nn->dense->activation_derivatives[i] = elu_prime;
                        } else if (strcmp(activationFunctionsStr[0], "tanh") == 0) {
                            nn->activation_functions_ref[i] = TANH;
                            nn->dense->activation_functions[i] = tan_h;
                            nn->dense->activation_derivatives[i] = tanh_prime;
                        } else if (strcmp(activationFunctionsStr[0], "softplus") == 0) {
                            nn->activation_functions_ref[i] = SOFTPLUS;
                            nn->dense->activation_functions[i] = softplus;
                            nn->dense->activation_derivatives[i] = softplus_prime;
                        } else fatal(DEFAULT_CONSOLE_WRITER, "unsupported or unrecognized activation function:", activationFunctionsStr[0]);
                    }
                } else {
                    for (int i=0; i<nn->network_num_layers-1; i++) {
                        if (strcmp(activationFunctionsStr[i], "sigmoid") == 0) {
                            nn->activation_functions_ref[i] = SIGMOID;
                            nn->dense->activation_functions[i] = sigmoid;
                            nn->dense->activation_derivatives[i] = sigmoid_prime;
                        } else if (strcmp(activationFunctionsStr[i], "relu") == 0) {
                            nn->activation_functions_ref[i] = RELU;
                            nn->dense->activation_functions[i] = relu;
                            nn->dense->activation_derivatives[i] = relu_prime;
                        } else if (strcmp(activationFunctionsStr[i], "leakyrelu") == 0) {
                            nn->activation_functions_ref[i] = LEAKY_RELU;
                            nn->dense->activation_functions[i] = leakyrelu;
                            nn->dense->activation_derivatives[i] = leakyrelu_prime;
                        } else if (strcmp(activationFunctionsStr[i], "elu") == 0) {
                            nn->activation_functions_ref[i] = ELU;
                            nn->dense->activation_functions[i] = elu;
                            nn->dense->activation_derivatives[i] = elu_prime;
                        } else if (strcmp(activationFunctionsStr[i], "tanh") == 0) {
                            nn->activation_functions_ref[i] = TANH;
                            nn->dense->activation_functions[i] = tan_h;
                            nn->dense->activation_derivatives[i] = tanh_prime;
                        } else if (strcmp(activationFunctionsStr[i], "softplus") == 0) {
                            nn->activation_functions_ref[i] = SOFTPLUS;
                            nn->dense->activation_functions[i] = softplus;
                            nn->dense->activation_derivatives[i] = softplus_prime;
                        } else if (strcmp(activationFunctionsStr[i], "softmax") == 0) {
                            // The sofmax function is only supported for the output units
                            if (i < nn->network_num_layers-2) {
                                fatal(DEFAULT_CONSOLE_WRITER, "the softmax function can't be used for the hiden units, only for the output units.");
                            }
                            nn->activation_functions_ref[i] = SOFTMAX;
                            nn->dense->activation_functions[i] = softmax;
                            nn->dense->activation_derivatives[i] = NULL;
                        } else fatal(DEFAULT_CONSOLE_WRITER, "unsupported or unrecognized activation function:", activationFunctionsStr[i]);
                    }
                }
                FOUND_ACTIVATIONS = 1;
                
            } else if (strcmp(field->key, "split") == 0) {
                unsigned int n;
                unsigned int len = 2;
                parse_argument(field->value,  field->key, nn->dense->parameters->split, &n, &len);
                if (n < 2) {
                    fatal(DEFAULT_CONSOLE_WRITER, " data splitting requires two values: one for training, one for testing/evaluation.");
                }
                FOUND_SPLIT = 1;
                
            } else if (strcmp(field->key, "classification") == 0) {
                unsigned int len = MAX_NUMBER_NETWORK_LAYERS;
                parse_argument(field->value, field->key, nn->dense->parameters->classifications, &nn->dense->parameters->num_classifications, &len);
                FOUND_CLASSIFICATION = 1;
                
            } else if (strcmp(field->key, "epochs") == 0) {
                nn->dense->parameters->epochs = atoi(field->value);
                
            } else if (strcmp(field->key, "batch_size") == 0) {
                nn->dense->parameters->mini_batch_size = atoi(field->value);
                
            } else if (strcmp(field->key, "learning_rate") == 0) {
                nn->dense->parameters->eta = strtof(field->value, NULL);
                
            } else if (strcmp(field->key, "l1_regularization") == 0) {
                if (FOUND_TOPOLOGY == 0) {
                    fatal(DEFAULT_CONSOLE_WRITER, "incorrect parameters definition order, the topology is not defined yet. ");
                }
                nn->dense->parameters->lambda = strtof(field->value, NULL);
                for (int i=0; i<nn->network_num_layers-1; i++) {
                    nn->regularizer[i] = nn->l1_regularizer;
                    FOUND_REGULARIZATION = 1;
                }
            } else if (strcmp(field->key, "l2_regularization") == 0) {
                if (FOUND_TOPOLOGY == 0) {
                    fatal(DEFAULT_CONSOLE_WRITER, "incorrect parameters definition order, the topology is not defined yet. ");
                }
                nn->dense->parameters->lambda = strtof(field->value, NULL);
                for (int i=0; i<nn->network_num_layers-1; i++) {
                    nn->regularizer[i] = nn->l2_regularizer;
                    FOUND_REGULARIZATION = 1;
                }
            } else if (strcmp(field->key, "gradient_descent_optimizer") == 0) {
                nn->dense->train->gradient_descent = (gradient_descent_optimizer *)malloc(sizeof(gradient_descent_optimizer));
                nn->dense->train->gradient_descent->learning_rate = strtof(field->value, NULL);
                nn->dense->parameters->eta = strtof(field->value, NULL);
                nn->dense->train->gradient_descent->minimize = gradient_descent_optimize;
                FOUND_OPTIMIZER = 1;
                
            } else if (strcmp(field->key, "momentum_optimizer") == 0) {
                nn->dense->train->momentum = (momentum_optimizer *)malloc(sizeof(momentum_optimizer));
                float result[2];
                unsigned int numberOfItems, len = 2;
                parse_argument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 2) fatal(DEFAULT_CONSOLE_WRITER, "the learming rate and the momentum coefficient should be given to the momentum optimizer.");
                nn->dense->train->momentum->learning_rate = result[0];
                nn->dense->train->momentum->momentum_coefficient = result[1];
                nn->dense->parameters->eta = result[0];
                nn->dense->train->momentum->minimize = momentum_optimize;
                FOUND_OPTIMIZER = 1;
            
            } else if (strcmp(field->key, "adagrad_optimizer") == 0) {
                nn->dense->train->ada_grad = (ada_grad_optimizer *)malloc(sizeof(ada_grad_optimizer));
                float result[2];
                unsigned int numberOfItems, len = 2;
                parse_argument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 2) fatal(DEFAULT_CONSOLE_WRITER, "the learming rate and a delta value should be given to the AdaGrad optimizer.");
                nn->dense->train->ada_grad->learning_rate = result[0];
                nn->dense->train->ada_grad->delta = result[1];
                nn->dense->parameters->eta = result[0];
                nn->dense->train->ada_grad->dense->cost_weight_derivative_squared_accumulated = NULL;
                nn->dense->train->ada_grad->dense->cost_bias_derivative_squared_accumulated = NULL;
                nn->dense->train->ada_grad->minimize = adam_optimize;
                FOUND_OPTIMIZER = 1;
            
            } else if (strcmp(field->key, "rmsprop_optimizer") == 0) {
                nn->dense->train->rms_prop = (rms_prop_optimizer *)malloc(sizeof(rms_prop_optimizer));
                float result[3];
                unsigned int numberOfItems, len = 3;
                parse_argument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 3) fatal(DEFAULT_CONSOLE_WRITER, "the learming rate, the decay rate and a delata value should be given to the RMSProp optimizer.");
                nn->dense->train->rms_prop->learning_rate = result[0];
                nn->dense->train->rms_prop->decay_rate = result[1];
                nn->dense->train->rms_prop->delta = result[2];
                nn->dense->parameters->eta = result[0];
                nn->dense->train->rms_prop->dense->cost_weight_derivative_squared_accumulated = NULL;
                nn->dense->train->rms_prop->dense->cost_bias_derivative_squared_accumulated = NULL;
                nn->dense->train->rms_prop->minimize = rms_prop_optimize;
                FOUND_OPTIMIZER = 1;
            
            } else if (strcmp(field->key, "adam_optimizer") == 0) {
                nn->dense->train->adam = (adam_optimizer *)malloc(sizeof(adam_optimizer));
                float result[4];
                unsigned int numberOfItems, len=4;
                parse_argument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 4) fatal(DEFAULT_CONSOLE_WRITER, "The step size, two decay rates and a delta value should be given to the Adam optimizer.");
                nn->dense->train->adam->time = 0;
                nn->dense->train->adam->step_size = result[0];
                nn->dense->train->adam->decay_rate1 = result[1];
                nn->dense->train->adam->decay_rate2 = result[2];
                nn->dense->train->adam->delta = result[3];
                nn->dense->parameters->eta = result[0];
                nn->dense->train->adam->dense->weights_biased_first_moment_estimate = NULL;
                nn->dense->train->adam->dense->weights_biased_second_moment_estimate = NULL;
                nn->dense->train->adam->dense->biases_biased_first_moment_estimate = NULL;
                nn->dense->train->adam->dense->biases_biased_second_moment_estimate = NULL;
                nn->dense->train->adam->minimize = adam_optimize;
                FOUND_OPTIMIZER = 1;
            }
            field = field->next;
        }
        pt = pt->next;
    }
    
    if (FOUND_DATA == 0) {
        fatal(DEFAULT_CONSOLE_WRITER, "missing data in parameters input.");
    }
    if (FOUND_TOPOLOGY == 0) {
        fatal(DEFAULT_CONSOLE_WRITER, "missing topology in parameters input.");
    }
    if (FOUND_ACTIVATIONS == 0) {
        for (int i=0; i<nn->network_num_layers-1; i++) {
            nn->dense->activation_functions[i] = sigmoid;
            nn->dense->activation_derivatives[i] = sigmoid_prime;
        }
    }
    if (FOUND_SPLIT == 0) {
        fatal(DEFAULT_CONSOLE_WRITER, "missing split in parameters input.");
    }
    if (FOUND_CLASSIFICATION == 0) {
        fatal(DEFAULT_CONSOLE_WRITER, "missing classification in parameters input.");
    }
    if (FOUND_REGULARIZATION == 0) {
        for (int i=0; i<nn->network_num_layers-1; i++) {
            nn->regularizer[i] = nn->l0_regularizer;
        }
    }
    if (FOUND_OPTIMIZER == 0) {
        fatal(DEFAULT_CONSOLE_WRITER, "missing optimizer in parameters input.");
    }
    
    return 0;
}
