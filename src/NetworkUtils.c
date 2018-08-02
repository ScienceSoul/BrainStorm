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

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// Tensor allocation //////////////////////////////////////////
//      This routine allocates a 1D, 2D, 3D, 4D or 5D tensor with the possibility to linearize several
//      tensors on a single data allocation (usufull for allocating accross network layers).
//      Xavier-He initialization is only possible for 2D and 4D tensors.
//      The tensor shape, rank and the flattening length are given inside the tensor_dict structure.
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void * _Nonnull tensor_create(void * _Nonnull self, tensor_dict tensor_dict) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    tensor *tensor_object = (tensor *)malloc(sizeof(tensor));
    memset(*tensor_object->shape, 1, (MAX_NUMBER_NETWORK_LAYERS*MAX_TENSOR_RANK*1));
    
    if (tensor_dict.init) {
        if (tensor_dict.init_strategy == NULL) fatal(DEFAULT_CONSOLE_WRITER, " initialization required but NULL strategy.");
    }
    
    int tensor_length = 0;
    for (int l=0; l<tensor_dict.flattening_length; l++) {
        int dim = 1;
        for (int i=0; i<tensor_dict.rank; i++) {
            dim = dim * tensor_dict.shape[l][i][0];
        }
        tensor_length = tensor_length + dim;
    }
    
    tensor_object->val = (float *)malloc(tensor_length*sizeof(float));
    memcpy(tensor_object->shape, tensor_dict.shape, (MAX_NUMBER_NETWORK_LAYERS*MAX_TENSOR_RANK*1));
    tensor_object->rank = tensor_dict.rank;
    fprintf(stdout, "%s: tensor allocation: allocate %f (MB)\n", DEFAULT_CONSOLE_WRITER, ((float)tensor_length*sizeof(float))/(float)(1024*1024));
    
    
    if (tensor_object->rank == 1) {
       if (tensor_dict.init) {
           int stride = 0;
           for (int l=0; l<tensor_dict.flattening_length; l++) {
               // One single tensor step
               int step = 1;
               for (int i=0; i<tensor_object->rank; i++) {
                   step = step * tensor_dict.shape[l][i][0];
               }
               
               int n = tensor_object->shape[l][0][0];
               for (int i=0; i<n; i++) {
                   tensor_object->val[stride+i] = randn(0.0f, 1.0f);
               }
               stride = stride + step;
           }
       } else {
           memset(tensor_object->val, 0.0f, tensor_length*sizeof(float));
       }
    } else if (tensor_object->rank == 2) {
        if (tensor_dict.init) {
            int stride = 0;
            for (int l=0; l<tensor_dict.flattening_length; l++) {
                // One single tensor step
                int step = 1;
                for (int i=0; i<tensor_object->rank; i++) {
                    step = step * tensor_dict.shape[l][i][0];
                }
                
                // The last two dimensions define the right most increments
                int indx = tensor_object->rank - 2;
                int m = tensor_object->shape[l][indx][0];
                int n = tensor_object->shape[l][indx+1][0];
                
                if (strcmp(tensor_dict.init_strategy, "default") == 0 || l == nn->network_num_layers-2) {
                    for (int i = 0; i<m; i++) {
                        for (int j=0; j<n; j++) {
                            tensor_object->val[stride+((i*n)+j)] = randn(0.0f, 1.0f) / sqrtf((float)n);
                        }
                    }
                } else if (strcmp(tensor_dict.init_strategy, "xavier-he") == 0  && l < nn->network_num_layers-2) {
                    float standard_deviation = 0.0f;
                    int n_inputs = 1;
                    for (int i=0; i<tensor_object->rank; i++) {
                         n_inputs =  n_inputs * tensor_dict.shape[l][i][0];
                    }
                    int n_outputs = 1;
                    for (int i=0; i<tensor_object->rank; i++) {
                        n_outputs =  n_outputs * tensor_dict.shape[l+1][i][0];
                    }
                    if (strcmp(nn->parameters->activationFunctions[l], "sigmoid") == 0) {
                        standard_deviation = sqrtf(2.0 / (float)(n_inputs + n_outputs));
                    } else if (strcmp(nn->parameters->activationFunctions[l], "tanh") == 0) {
                        standard_deviation = powf((2/(float)(n_inputs + n_outputs)), (1.0/4.0));
                    } else if (strcmp(nn->parameters->activationFunctions[l], "relu") == 0 ||
                               strcmp(nn->parameters->activationFunctions[l], "leakyrelu") == 0 ||
                               strcmp(nn->parameters->activationFunctions[l], "elu") == 0) {
                        standard_deviation = sqrtf(2.0f) * sqrtf(2.0 / (float)(n_inputs + n_outputs));
                    }
                    for (int i = 0; i<m; i++) {
                        for (int j=0; j<n; j++) {
                            tensor_object->val[stride+((i*n)+j)] = randn(0.0f, standard_deviation);
                        }
                    }
                }
                stride = stride + step;
            }
        } else {
            memset(tensor_object->val, 0.0f, tensor_length*sizeof(float));
        }
    } else if (tensor_object->rank == 3) {
        if (tensor_dict.init) {
            int stride1 = 0;
            for (int l=0; l<tensor_dict.flattening_length; l++) {
                // One single tensor step
                int step = 1;
                for (int i=0; i<tensor_object->rank; i++) {
                    step = step * tensor_dict.shape[l][i][0];
                }
                
                // The last two dimensions define the right most increments
                int indx = tensor_object->rank - 2;
                int m = tensor_object->shape[l][indx][0];
                int n = tensor_object->shape[l][indx+1][0];
                
                if (strcmp(tensor_dict.init_strategy, "default") == 0 || l == nn->network_num_layers-2) {
                    int stride2 = 0;
                    for (int k=0; k<tensor_dict.shape[l][0][0]; k++) {
                        for (int i = 0; i<m; i++) {
                            for (int j=0; j<n; j++) {
                                tensor_object->val[stride1+(stride2+((i*n)+j))] = randn(0.0f, 1.0f) / sqrtf((float)n);
                            }
                        }
                        stride2 = stride2 + (m * n);
                    }
                } else if (strcmp(tensor_dict.init_strategy, "xavier-he") == 0  && l < nn->network_num_layers-2) {
                    fatal(DEFAULT_CONSOLE_WRITER, "Xavier-He initialization is not supported for 3D tensors. Only available for 1D and 4D tensors (aka fully-connected and convolution layers).");
                }
                stride1 = stride1 + step;
            }
        } else {
            memset(tensor_object->val, 0.0f, tensor_length*sizeof(float));
        }
    } else if (tensor_object->rank == 4) {
        if (tensor_dict.init) {
            int stride1 = 0;
            for (int l=0; l<tensor_dict.flattening_length; l++) {
                // One single tensor step
                int step = 1;
                for (int i=0; i<tensor_object->rank; i++) {
                    step = step * tensor_dict.shape[l][i][0];
                }
                
                // The last two dimensions define the right most increments
                int indx = tensor_object->rank - 2;
                int m = tensor_object->shape[l][indx][0];
                int n = tensor_object->shape[l][indx+1][0];
                
                if (strcmp(tensor_dict.init_strategy, "default") == 0 || l == nn->network_num_layers-2) {
                    int stride2 = 0;
                    for (int k=0; k<tensor_dict.shape[l][0][0]; k++) {
                        int stride3 = 0;
                        for (int ll=0; ll<tensor_dict.shape[l][1][0]; ll++) {
                            for (int i = 0; i<m; i++) {
                                for (int j=0; j<n; j++) {
                                    tensor_object->val[stride1+(stride2+(stride3+((i*n)+j)))] = randn(0.0f, 1.0f) / sqrtf((float)n);
                                }
                            }
                            stride3 = stride3 + (m * n);
                        }
                        stride2 = stride2 + (tensor_dict.shape[l][1][0] * m * n);
                    }
                } else if (strcmp(tensor_dict.init_strategy, "xavier-he") == 0  && l < nn->network_num_layers-2) {
                     int stride2 = 0;
                     for (int k=0; k<tensor_dict.shape[l][0][0]; k++) {
                         int stride3 = 0;
                         for (int ll=0; ll<tensor_dict.shape[l][1][0]; ll++) {
                             float standard_deviation = 0.0f;
                             int n_inputs = tensor_dict.shape[l][0][0] * tensor_dict.shape[l][1][0];
                             int n_outputs = tensor_dict.shape[l][1][0];
                             if (strcmp(nn->parameters->activationFunctions[l], "sigmoid") == 0) {
                                 standard_deviation = sqrtf(2.0 / (float)(n_inputs + n_outputs));
                             } else if (strcmp(nn->parameters->activationFunctions[l], "tanh") == 0) {
                                 standard_deviation = powf((2/(float)(n_inputs + n_outputs)), (1.0/4.0));
                             } else if (strcmp(nn->parameters->activationFunctions[l], "relu") == 0 ||
                                        strcmp(nn->parameters->activationFunctions[l], "leakyrelu") == 0 ||
                                        strcmp(nn->parameters->activationFunctions[l], "elu") == 0) {
                                 standard_deviation = sqrtf(2.0f) * sqrtf(2.0 / (float)(n_inputs + n_outputs));
                             }
                             for (int i = 0; i<m; i++) {
                                 for (int j=0; j<n; j++) {
                                     tensor_object->val[stride1+(stride2+(stride3+((i*n)+j)))] = randn(0.0f, standard_deviation);
                                 }
                             }
                             stride3 = stride3 + (m * n);
                         }
                         stride2 = stride2 + (tensor_dict.shape[l][1][0] * m * n);
                     }
                }
                stride1 = stride1 + step;
            }
        } else {
            memset(tensor_object->val, 0.0f, tensor_length*sizeof(float));
        }
    } else if (tensor_object->rank == 5) {
         if (tensor_dict.init) {
             int stride1 = 0;
             for (int l=0; l<tensor_dict.flattening_length; l++) {
                 // One single tensor step
                 int step = 1;
                 for (int i=0; i<tensor_object->rank; i++) {
                     step = step * tensor_dict.shape[l][i][0];
                 }
                 
                 // The last two dimensions define the right most increments
                 int indx = tensor_object->rank - 2;
                 int m = tensor_object->shape[l][indx][0];
                 int n = tensor_object->shape[l][indx+1][0];
                 
                 if (strcmp(tensor_dict.init_strategy, "default") == 0 || l == nn->network_num_layers-2) {
                     int stride2 = 0;
                     for (int k=0; k<tensor_dict.shape[l][0][0]; k++) {
                         int stride3 = 0;
                         for (int ll=0; ll<tensor_dict.shape[l][1][0]; ll++) {
                             int stride4 = 0;
                             for (int ll=0; ll<tensor_dict.shape[l][2][0]; ll++) {
                                 for (int i = 0; i<m; i++) {
                                     for (int j=0; j<n; j++) {
                                         tensor_object->val[stride1+(stride2+(stride3+(stride4+((i*n)+j))))] = randn(0.0f, 1.0f) / sqrtf((float)n);
                                     }
                                 }
                                 stride4 = stride4 + (m * n);
                             }
                             stride3 = stride3 + (tensor_dict.shape[l][2][0] * m *n);
                         }
                         stride2 = stride2 + (tensor_dict.shape[l][1][0] * tensor_dict.shape[l][2][0] * m * n);
                     }
                 
                 } else if (strcmp(tensor_dict.init_strategy, "xavier-he") == 0  && l < nn->network_num_layers-2) {
                     fatal(DEFAULT_CONSOLE_WRITER, "Xavier-He initialization is not supported for 5D tensors. Only available for 1D and 4D tensors (aka fully-connected and convolution layers).");
                 }
                 stride1 = stride1 + step;
             }
         } else {
             memset(tensor_object->val, 0.0f, tensor_length*sizeof(float));
         }
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "Tensors with rank > 5 are not supported.");
    }
    
    return (void *)tensor_object;
}

int loadParametersFromImputFile(void * _Nonnull self, const char * _Nonnull paraFile) {
    
    definition *definitions = NULL;
    
    fprintf(stdout, "%s: load the network and its input parameters...\n", DEFAULT_CONSOLE_WRITER);
    
    definitions = getDefinitions(self, paraFile, "define");
    if (definitions == NULL) {
        fatal(DEFAULT_CONSOLE_WRITER, "problem finding any parameter definition.");
    }
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
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
                if (strcmp(field->key, nn->parameters->supported_parameters[i]) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) fatal(DEFAULT_CONSOLE_WRITER, "key for parameter not recognized:", field->key);
            
            if (strcmp(field->key, "data_name") == 0) {
                strcpy(nn->parameters->dataName, field->value);
                
            } else if (strcmp(field->key, "data") == 0) {
                strcpy(nn->parameters->data, field->value);
                FOUND_DATA = 1;
                
            } else if (strcmp(field->key, "topology") == 0) {
                unsigned int len = MAX_NUMBER_NETWORK_LAYERS;
                parseArgument(field->value, field->key, nn->parameters->topology, &nn->network_num_layers, &len);
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
                
                unsigned int len = MAX_NUMBER_NETWORK_LAYERS;
                parseArgument(field->value, field->key, nn->parameters->activationFunctions, &nn->parameters->numberOfActivationFunctions, &len);
                
                if (nn->parameters->numberOfActivationFunctions > 1 && nn->parameters->numberOfActivationFunctions < nn->network_num_layers-1) {
                    fatal(DEFAULT_CONSOLE_WRITER, "the number of activation functions in parameters is too low. Can't resolve how to use the provided activations. ");
                }
                if (nn->parameters->numberOfActivationFunctions > nn->network_num_layers-1) {
                    fprintf(stdout, "%s: too many activation functions given to network. Will ignore the extra ones.\n", DEFAULT_CONSOLE_WRITER);
                }
                
                if (nn->parameters->numberOfActivationFunctions == 1) {
                    if (strcmp(nn->parameters->activationFunctions[0], "softmax") == 0) {
                        fatal(DEFAULT_CONSOLE_WRITER, "the softmax function can only be used for the output units, not for the entire network.");
                    }
                    for (int i=0; i<nn->network_num_layers-1; i++) {
                        if (strcmp(nn->parameters->activationFunctions[0], "sigmoid") == 0) {
                            nn->activationFunctions[i] = sigmoid;
                            nn->activationDerivatives[i] = sigmoidPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[0], "relu") == 0) {
                            nn->activationFunctions[i] = relu;
                            nn->activationDerivatives[i] = reluPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[0], "leakyrelu") == 0) {
                            nn->activationFunctions[i] = leakyrelu;
                            nn->activationDerivatives[i] = leakyreluPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[0], "elu") == 0) {
                            nn->activationFunctions[i] = elu;
                            nn->activationDerivatives[i] = eluPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[0], "tanh") == 0) {
                            nn->activationFunctions[i] = tan_h;
                            nn->activationDerivatives[i] = tanhPrime;
                        } else fatal(DEFAULT_CONSOLE_WRITER, "unsupported or unrecognized activation function:", nn->parameters->activationFunctions[0]);
                    }
                } else {
                    for (int i=0; i<nn->network_num_layers-1; i++) {
                        if (strcmp(nn->parameters->activationFunctions[i], "sigmoid") == 0) {
                            nn->activationFunctions[i] = sigmoid;
                            nn->activationDerivatives[i] = sigmoidPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[i], "relu") == 0) {
                            nn->activationFunctions[i] = relu;
                            nn->activationDerivatives[i] = reluPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[i], "leakyrelu") == 0) {
                            nn->activationFunctions[i] = leakyrelu;
                            nn->activationDerivatives[i] = leakyreluPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[i], "elu") == 0) {
                            nn->activationFunctions[i] = elu;
                            nn->activationDerivatives[i] = eluPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[i], "tanh") == 0) {
                            nn->activationFunctions[i] = tan_h;
                            nn->activationDerivatives[i] = tanhPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[i], "softmax") == 0) {
                            // The sofmax function is only supported for the output units
                            if (i < nn->network_num_layers-2) {
                                fatal(DEFAULT_CONSOLE_WRITER, "the softmax function can't be used for the hiden units, only for the output units.");
                            }
                            nn->activationFunctions[i] = softmax;
                            nn->activationDerivatives[i] = NULL;
                        } else fatal(DEFAULT_CONSOLE_WRITER, "unsupported or unrecognized activation function:", nn->parameters->activationFunctions[i]);
                    }
                }
                FOUND_ACTIVATIONS = 1;
                
            } else if (strcmp(field->key, "split") == 0) {
                unsigned int n;
                unsigned int len = 2;
                parseArgument(field->value,  field->key, nn->parameters->split, &n, &len);
                if (n < 2) {
                    fatal(DEFAULT_CONSOLE_WRITER, " data splitting requires two values: one for training, one for testing/evaluation.");
                }
                FOUND_SPLIT = 1;
                
            } else if (strcmp(field->key, "classification") == 0) {
                unsigned int len = MAX_NUMBER_NETWORK_LAYERS;
                parseArgument(field->value, field->key, nn->parameters->classifications, &nn->parameters->numberOfClassifications, &len);
                FOUND_CLASSIFICATION = 1;
                
            } else if (strcmp(field->key, "epochs") == 0) {
                nn->parameters->epochs = atoi(field->value);
                
            } else if (strcmp(field->key, "batch_size") == 0) {
                nn->parameters->miniBatchSize = atoi(field->value);
                
            } else if (strcmp(field->key, "learning_rate") == 0) {
                nn->parameters->eta = strtof(field->value, NULL);
                
            } else if (strcmp(field->key, "l1_regularization") == 0) {
                if (FOUND_TOPOLOGY == 0) {
                    fatal(DEFAULT_CONSOLE_WRITER, "incorrect parameters definition order, the topology is not defined yet. ");
                }
                nn->parameters->lambda = strtof(field->value, NULL);
                for (int i=0; i<nn->network_num_layers-1; i++) {
                    nn->regularizer[i] = nn->l1_regularizer;
                    FOUND_REGULARIZATION = 1;
                }
            } else if (strcmp(field->key, "l2_regularization") == 0) {
                if (FOUND_TOPOLOGY == 0) {
                    fatal(DEFAULT_CONSOLE_WRITER, "incorrect parameters definition order, the topology is not defined yet. ");
                }
                nn->parameters->lambda = strtof(field->value, NULL);
                for (int i=0; i<nn->network_num_layers-1; i++) {
                    nn->regularizer[i] = nn->l2_regularizer;
                    FOUND_REGULARIZATION = 1;
                }
            } else if (strcmp(field->key, "gradient_descent_optimizer") == 0) {
                nn->dense->train->gradient_descent = (GradientDescentOptimizer *)malloc(sizeof(GradientDescentOptimizer));
                nn->dense->train->gradient_descent->learning_rate = strtof(field->value, NULL);
                nn->parameters->eta = strtof(field->value, NULL);
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
                nn->parameters->eta = result[0];
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
                nn->parameters->eta = result[0];
                nn->dense->train->ada_grad->costWeightDerivativeSquaredAccumulated = NULL;
                nn->dense->train->ada_grad->costBiasDerivativeSquaredAccumulated = NULL;
                nn->dense->train->ada_grad->minimize = adamOptimizer;
                FOUND_OPTIMIZER = 1;
            
            } else if (strcmp(field->key, "rmsprop_optimizer") == 0) {
                nn->dense->train->rms_prop = (RMSPropOptimizer *)malloc(sizeof(RMSPropOptimizer));
                float result[3];
                unsigned int numberOfItems, len = 3;
                parseArgument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 3) fatal(DEFAULT_CONSOLE_WRITER, "the learming rate, the decay rate and a delata value should be given to the RMSProp optimizer.");
                nn->dense->train->rms_prop->learning_rate = result[0];
                nn->dense->train->rms_prop->decayRate = result[1];
                nn->dense->train->rms_prop->delta = result[2];
                nn->parameters->eta = result[0];
                nn->dense->train->rms_prop->costWeightDerivativeSquaredAccumulated = NULL;
                nn->dense->train->rms_prop->costBiasDerivativeSquaredAccumulated = NULL;
                nn->dense->train->rms_prop->minimize = rmsPropOptimizer;
                FOUND_OPTIMIZER = 1;
            
            } else if (strcmp(field->key, "adam_optimizer") == 0) {
                nn->dense->train->adam = (AdamOptimizer *)malloc(sizeof(AdamOptimizer));
                float result[4];
                unsigned int numberOfItems, len=4;
                parseArgument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 4) fatal(DEFAULT_CONSOLE_WRITER, "The step size, two decay rates and a delta value should be given to the Adam optimizer.");
                nn->dense->train->adam->time = 0;
                nn->dense->train->adam->stepSize = result[0];
                nn->dense->train->adam->decayRate1 = result[1];
                nn->dense->train->adam->decayRate2 = result[2];
                nn->dense->train->adam->delta = result[3];
                nn->parameters->eta = result[0];
                nn->dense->train->adam->weightsBiasedFirstMomentEstimate = NULL;
                nn->dense->train->adam->weightsBiasedSecondMomentEstimate = NULL;
                nn->dense->train->adam->biasesBiasedFirstMomentEstimate = NULL;
                nn->dense->train->adam->biasesBiasedSecondMomentEstimate = NULL;
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
            nn->activationFunctions[i] = sigmoid;
            nn->activationDerivatives[i] = sigmoidPrime;
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
