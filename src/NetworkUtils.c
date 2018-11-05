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
void * _Nullable tensor_create(void * _Nonnull self, tensor_dict tensor_dict) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
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
    memcpy(tensor_object->shape, tensor_dict.shape, (MAX_NUMBER_NETWORK_LAYERS*MAX_TENSOR_RANK*1));
    tensor_object->rank = tensor_dict.rank;
    fprintf(stdout, "%s: tensor allocation: allocate %f (MB)\n", DEFAULT_CONSOLE_WRITER, ((float)tensor_length*sizeof(float))/(float)(1024*1024));
    
    if (tensor_dict.init_neural_params) {
        int offset = 0;
        for (int l=0; l<tensor_dict.flattening_length; l++) {
            // One single tensor step
            int step = 1;
            for (int i=0; i<tensor_object->rank; i++) {
                step = step * tensor_dict.shape[l][i][0];
            }
            
            if (nn->kernelInitializers[l] == xavier_initializer) {
                nn->kernelInitializers[l]((void *)tensor_object, NULL, NULL, NULL, l, offset, NULL);
            } else if (nn->kernelInitializers[l] == variance_scaling_initializer) {
                bool uniform = false;
                nn->kernelInitializers[l]((void *)tensor_object, NULL, "FAN_IN", &uniform, l, offset, NULL);
            } else if (nn->kernelInitializers[l] == random_normal_initializer) {
                nn->kernelInitializers[l]((void *)tensor_object, NULL, NULL, NULL, l, offset, NULL);
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
    
    *(dict) = (tensor_dict){.init_neural_params=false, .init_with_value=false, .full_connected=false, .flattening_length=1, .rank =0, .init_value=0.0f};
    memset(**dict->shape, 0, (MAX_NUMBER_NETWORK_LAYERS*MAX_TENSOR_RANK)*sizeof(int));
    return dict;
}

int loadParametersFromImputFile(void * _Nonnull self, const char * _Nonnull paraFile) {
    
    definition *definitions = NULL;
    
    fprintf(stdout, "%s: load the network and its input parameters...\n", DEFAULT_CONSOLE_WRITER);
    
    definitions = getDefinitions(self, paraFile, "define");
    if (definitions == NULL) {
        fatal(DEFAULT_CONSOLE_WRITER, "problem finding any parameter definition.");
    }
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
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
                strcpy(nn->dataName, field->value);
                
            } else if (strcmp(field->key, "data") == 0) {
                strcpy(nn->dataPath, field->value);
                FOUND_DATA = 1;
                
            } else if (strcmp(field->key, "topology") == 0) {
                unsigned int len = MAX_NUMBER_NETWORK_LAYERS;
                parseArgument(field->value, field->key, nn->dense->parameters->topology, &nn->network_num_layers, &len);
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
                parseArgument(field->value, field->key, activationFunctionsStr, &nn->num_activation_functions, &len);
                
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
                            nn->activationFunctionsRef[i] = SIGMOID;
                            nn->dense->activationFunctions[i] = sigmoid;
                            nn->dense->activationDerivatives[i] = sigmoidPrime;
                        } else if (strcmp(activationFunctionsStr[0], "relu") == 0) {
                            nn->activationFunctionsRef[i] = RELU;
                            nn->dense->activationFunctions[i] = relu;
                            nn->dense->activationDerivatives[i] = reluPrime;
                        } else if (strcmp(activationFunctionsStr[0], "leakyrelu") == 0) {
                            nn->activationFunctionsRef[i] = LEAKY_RELU;
                            nn->dense->activationFunctions[i] = leakyrelu;
                            nn->dense->activationDerivatives[i] = leakyreluPrime;
                        } else if (strcmp(activationFunctionsStr[0], "elu") == 0) {
                            nn->activationFunctionsRef[i] = ELU;
                            nn->dense->activationFunctions[i] = elu;
                            nn->dense->activationDerivatives[i] = eluPrime;
                        } else if (strcmp(activationFunctionsStr[0], "tanh") == 0) {
                            nn->activationFunctionsRef[i] = TANH;
                            nn->dense->activationFunctions[i] = tan_h;
                            nn->dense->activationDerivatives[i] = tanhPrime;
                        } else fatal(DEFAULT_CONSOLE_WRITER, "unsupported or unrecognized activation function:", activationFunctionsStr[0]);
                    }
                } else {
                    for (int i=0; i<nn->network_num_layers-1; i++) {
                        if (strcmp(activationFunctionsStr[i], "sigmoid") == 0) {
                            nn->activationFunctionsRef[i] = SIGMOID;
                            nn->dense->activationFunctions[i] = sigmoid;
                            nn->dense->activationDerivatives[i] = sigmoidPrime;
                        } else if (strcmp(activationFunctionsStr[i], "relu") == 0) {
                            nn->activationFunctionsRef[i] = RELU;
                            nn->dense->activationFunctions[i] = relu;
                            nn->dense->activationDerivatives[i] = reluPrime;
                        } else if (strcmp(activationFunctionsStr[i], "leakyrelu") == 0) {
                            nn->activationFunctionsRef[i] = LEAKY_RELU;
                            nn->dense->activationFunctions[i] = leakyrelu;
                            nn->dense->activationDerivatives[i] = leakyreluPrime;
                        } else if (strcmp(activationFunctionsStr[i], "elu") == 0) {
                            nn->activationFunctionsRef[i] = ELU;
                            nn->dense->activationFunctions[i] = elu;
                            nn->dense->activationDerivatives[i] = eluPrime;
                        } else if (strcmp(activationFunctionsStr[i], "tanh") == 0) {
                            nn->activationFunctionsRef[i] = TANH;
                            nn->dense->activationFunctions[i] = tan_h;
                            nn->dense->activationDerivatives[i] = tanhPrime;
                        } else if (strcmp(activationFunctionsStr[i], "softmax") == 0) {
                            // The sofmax function is only supported for the output units
                            if (i < nn->network_num_layers-2) {
                                fatal(DEFAULT_CONSOLE_WRITER, "the softmax function can't be used for the hiden units, only for the output units.");
                            }
                            nn->activationFunctionsRef[i] = SOFTMAX;
                            nn->dense->activationFunctions[i] = softmax;
                            nn->dense->activationDerivatives[i] = NULL;
                        } else fatal(DEFAULT_CONSOLE_WRITER, "unsupported or unrecognized activation function:", activationFunctionsStr[i]);
                    }
                }
                FOUND_ACTIVATIONS = 1;
                
            } else if (strcmp(field->key, "split") == 0) {
                unsigned int n;
                unsigned int len = 2;
                parseArgument(field->value,  field->key, nn->dense->parameters->split, &n, &len);
                if (n < 2) {
                    fatal(DEFAULT_CONSOLE_WRITER, " data splitting requires two values: one for training, one for testing/evaluation.");
                }
                FOUND_SPLIT = 1;
                
            } else if (strcmp(field->key, "classification") == 0) {
                unsigned int len = MAX_NUMBER_NETWORK_LAYERS;
                parseArgument(field->value, field->key, nn->dense->parameters->classifications, &nn->dense->parameters->numberOfClassifications, &len);
                FOUND_CLASSIFICATION = 1;
                
            } else if (strcmp(field->key, "epochs") == 0) {
                nn->dense->parameters->epochs = atoi(field->value);
                
            } else if (strcmp(field->key, "batch_size") == 0) {
                nn->dense->parameters->miniBatchSize = atoi(field->value);
                
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
                nn->dense->train->gradient_descent = (GradientDescentOptimizer *)malloc(sizeof(GradientDescentOptimizer));
                nn->dense->train->gradient_descent->learning_rate = strtof(field->value, NULL);
                nn->dense->parameters->eta = strtof(field->value, NULL);
                nn->dense->train->gradient_descent->minimize = gradientDescentOptimizer;
                FOUND_OPTIMIZER = 1;
                
            } else if (strcmp(field->key, "momentum_optimizer") == 0) {
                nn->dense->train->momentum = (MomentumOptimizer *)malloc(sizeof(MomentumOptimizer));
                float result[2];
                unsigned int numberOfItems, len = 2;
                parseArgument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 2) fatal(DEFAULT_CONSOLE_WRITER, "the learming rate and the momentum coefficient should be given to the momentum optimizer.");
                nn->dense->train->momentum->learning_rate = result[0];
                nn->dense->train->momentum->momentum_coefficient = result[1];
                nn->dense->parameters->eta = result[0];
                nn->dense->train->momentum->minimize = momentumOptimizer;
                FOUND_OPTIMIZER = 1;
            
            } else if (strcmp(field->key, "adagrad_optimizer") == 0) {
                nn->dense->train->ada_grad = (AdaGradOptimizer *)malloc(sizeof(AdaGradOptimizer));
                float result[2];
                unsigned int numberOfItems, len = 2;
                parseArgument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 2) fatal(DEFAULT_CONSOLE_WRITER, "the learming rate and a delta value should be given to the AdaGrad optimizer.");
                nn->dense->train->ada_grad->learning_rate = result[0];
                nn->dense->train->ada_grad->delta = result[1];
                nn->dense->parameters->eta = result[0];
                nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated = NULL;
                nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated = NULL;
                nn->dense->train->ada_grad->minimize = adamOptimizer;
                FOUND_OPTIMIZER = 1;
            
            } else if (strcmp(field->key, "rmsprop_optimizer") == 0) {
                nn->dense->train->rms_prop = (RMSPropOptimizer *)malloc(sizeof(RMSPropOptimizer));
                float result[3];
                unsigned int numberOfItems, len = 3;
                parseArgument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 3) fatal(DEFAULT_CONSOLE_WRITER, "the learming rate, the decay rate and a delata value should be given to the RMSProp optimizer.");
                nn->dense->train->rms_prop->learning_rate = result[0];
                nn->dense->train->rms_prop->decay_rate = result[1];
                nn->dense->train->rms_prop->delta = result[2];
                nn->dense->parameters->eta = result[0];
                nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated = NULL;
                nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated = NULL;
                nn->dense->train->rms_prop->minimize = rmsPropOptimizer;
                FOUND_OPTIMIZER = 1;
            
            } else if (strcmp(field->key, "adam_optimizer") == 0) {
                nn->dense->train->adam = (AdamOptimizer *)malloc(sizeof(AdamOptimizer));
                float result[4];
                unsigned int numberOfItems, len=4;
                parseArgument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 4) fatal(DEFAULT_CONSOLE_WRITER, "The step size, two decay rates and a delta value should be given to the Adam optimizer.");
                nn->dense->train->adam->time = 0;
                nn->dense->train->adam->step_size = result[0];
                nn->dense->train->adam->decay_rate1 = result[1];
                nn->dense->train->adam->decay_rate2 = result[2];
                nn->dense->train->adam->delta = result[3];
                nn->dense->parameters->eta = result[0];
                nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate = NULL;
                nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate = NULL;
                nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate = NULL;
                nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate = NULL;
                nn->dense->train->adam->minimize = adamOptimizer;
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
            nn->dense->activationFunctions[i] = sigmoid;
            nn->dense->activationDerivatives[i] = sigmoidPrime;
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
