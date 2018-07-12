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

void * _Nonnull allocateActivationNode(void) {
    activationNode *list = (activationNode *)malloc(sizeof(activationNode));
    *list = (activationNode){.n=0, .a=NULL, .next=NULL, .previous=NULL};
    return (void *)list;
}

void * _Nonnull allocateAffineTransformationNode(void) {
    affineTransformationNode *list = (affineTransformationNode *)malloc(sizeof(affineTransformationNode));
    *list = (affineTransformationNode){.n=0, .z=NULL, .next=NULL, .previous=NULL};
    return (void *)list;
}

void * _Nonnull allocateCostWeightDerivativeNode(void) {
    costWeightDerivativeNode *list = (costWeightDerivativeNode *)malloc(sizeof(costWeightDerivativeNode));
    *list = (costWeightDerivativeNode){.m=0, .n=0, .dcdw=NULL, .next=NULL, .previous=NULL};
    return (void *)list;
}

void * _Nonnull allocateCostBiaseDerivativeNode(void) {
    costBiaseDerivativeNode *list = (costBiaseDerivativeNode *)malloc(sizeof(costBiaseDerivativeNode));
    *list = (costBiaseDerivativeNode){.n=0, .dcdb=NULL, .next=NULL, .previous=NULL};
    return (void *)list;
}

//
//  Allocate and initialize serialized matrices
//  They are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1 over the square root of the number of
//  weights/velocities connecting to the same neuron.
//
float * _Nonnull initMatrices(void * _Nonnull self, bool init, char * _Nonnull strategy) {
    
    NeuralNetwork* nn= (NeuralNetwork *)self;
    
    if (init) {
        if (strategy == NULL) fatal(DEFAULT_CONSOLE_WRITER, " initialization required but NULL strategy.");
    }
    
    int dim = 0;
    for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
        dim = dim + (nn->parameters->topology[l+1]*nn->parameters->topology[l]);
    }
    fprintf(stdout, "%s: matrices allocation: allocate %f (MB)\n", DEFAULT_CONSOLE_WRITER, ((float)dim*sizeof(float))/(float)(1024*1024));
    float *matrices = (float *)malloc(dim*sizeof(float));
    
    if (init) {
        int stride = 0;
        for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
            int m = nn->parameters->topology[l+1];
            int n = nn->parameters->topology[l];
            // Output layer always initialized the same way
            if (strcmp(strategy, "default") == 0 || l == nn->parameters->numberOfLayers-2) {
                for (int i = 0; i<m; i++) {
                    for (int j=0; j<n; j++) {
                        matrices[stride+((i*n)+j)] = randn(0.0f, 1.0f) / sqrtf((float)n);
                    }
                }
            } else if (strcmp(strategy, "xavier-he") == 0  && l < nn->parameters->numberOfLayers-2) { // xavier-he only used for hidden layers
                float standard_deviation = 0.0f;
                int n_inputs = m * n;
                int n_outputs = m * nn->parameters->topology[l+2];
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
                        matrices[stride+((i*n)+j)] = randn(0.0f, standard_deviation);
                    }
                }
            } else {
                fatal(DEFAULT_CONSOLE_WRITER, "unrecognized initializer in matrices.");
            }
            stride = stride + (m * n);
        }
    } else {
        memset(matrices, 0.0f, dim*sizeof(float));
    }
    
    return matrices;
}

//
//  Allocate and initialize serialized vectors
//  They are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1.
//
float * _Nonnull initVectors(void * _Nonnull self, bool init, char * _Nonnull strategy) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    if (init) {
        if (strategy == NULL) fatal(DEFAULT_CONSOLE_WRITER, " initialization required but NULL strategy.");
    }
    
    int dim = 0;
    for (int l=1; l<nn->parameters->numberOfLayers; l++) {
        dim  = dim + nn->parameters->topology[l];
    }
    fprintf(stdout, "%s: vectors allocation: allocate %f (MB)\n", DEFAULT_CONSOLE_WRITER, ((float)dim*sizeof(float))/(float)(1024*1024));
    float *vectors = (float*)malloc(dim*sizeof(float));
    
    if (init) {
        int stride = 0;
        for (int l=1; l<nn->parameters->numberOfLayers; l++) {
            int n = nn->parameters->topology[l];
            for (int i = 0; i<n; i++) {
                vectors[stride+i] = randn(0.0f, 1.0f);
            }
            stride = stride + n;
        }
    } else {
        memset(vectors, 0.0f, dim*sizeof(float));
    }
    
    return vectors;
}

float * _Nonnull tensor(void * _Nonnull self, tensor_dict tensor_dict) {
    
    float *tensor = NULL;
    if (tensor_dict.rank == 1) {
        tensor = initVectors(self, tensor_dict.init, tensor_dict.init_stategy);
    } else if (tensor_dict.rank == 2) {
        tensor = initMatrices(self, tensor_dict.init, tensor_dict.init_stategy);
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "creating a tensor with rank > 2 is not currently supported.");
    }
    return tensor;
}

//
//  Create the network activations
//  Return a pointer to the head of the linked list
//
void * _Nonnull initNetworkActivations(int * _Nonnull ntLayers, unsigned int numberOfLayers) {
    
    activationNode *activationsList = (activationNode *)allocateActivationNode();
    
    // The first activation node (i.e., layer)
    activationsList->a = floatvec(0, ntLayers[0]-1);
    activationsList->n = ntLayers[0];
    // The rest of the activation nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    activationNode *aNodePt = activationsList;
    while (k <= numberOfLayers-1) {
        activationNode *newNode = (activationNode *)allocateActivationNode();
        newNode->a = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = aNodePt;
        aNodePt->next = newNode;
        aNodePt = newNode;
        k++;
        idx++;
    }
    
    aNodePt = activationsList;
    while (aNodePt != NULL) {
        memset(aNodePt->a, 0.0f, aNodePt->n*sizeof(float));
        aNodePt = aNodePt->next;
    }
    
    return (void *)activationsList;
}

//
//  Create the network affine transformations
//  Return a pointer to the head of the linked list
//
void * _Nonnull initNetworkAffineTransformations(int * _Nonnull ntLayers, unsigned int numberOfLayers) {
    
    affineTransformationNode *zsList = (affineTransformationNode *)allocateAffineTransformationNode();
    
    // The first z node (i.e., layer)
    zsList->z = floatvec(0, ntLayers[0]-1);
    zsList->n = ntLayers[0];
    // The rest of the z nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    affineTransformationNode *zNodePt = zsList;
    while (k <= numberOfLayers-1) {
        affineTransformationNode *newNode = (affineTransformationNode *)allocateAffineTransformationNode();
        newNode->z = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = zNodePt;
        zNodePt->next = newNode;
        zNodePt = newNode;
        k++;
        idx++;
    }
    
    zNodePt = zsList;
    while (zNodePt != NULL) {
        memset(zNodePt->z, 0.0f, zNodePt->n*sizeof(float));
        zNodePt = zNodePt->next;
    }
    
    return (void *)zsList;
}

//
//  Create the network derivatives of the cost function with respect to the weights
//  Return a pointer to the head of the linked list
//
void * _Nonnull initNetworkCostWeightDerivatives(int * _Nonnull ntLayers, unsigned int numberOfLayers) {
    
    costWeightDerivativeNode *dcdwList = (costWeightDerivativeNode *)allocateCostWeightDerivativeNode();
    
    // The first weight node (i.e., layer)
    dcdwList->dcdw = floatmatrix(0, ntLayers[1]-1, 0, ntLayers[0]-1);
    dcdwList->m = ntLayers[1];
    dcdwList->n = ntLayers[0];
    // The rest of the weight nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    costWeightDerivativeNode *dcdwNodePt = dcdwList;
    while (k < numberOfLayers-1) {
        costWeightDerivativeNode *newNode = (costWeightDerivativeNode *)allocateCostWeightDerivativeNode();
        newNode->dcdw = floatmatrix(0, ntLayers[idx+1]-1, 0, ntLayers[idx]-1);
        newNode->m = ntLayers[idx+1];
        newNode->n = ntLayers[idx];
        newNode->previous = dcdwNodePt;
        dcdwNodePt->next = newNode;
        dcdwNodePt = newNode;
        k++;
        idx++;
    }
    
    return (void *)dcdwList;
}

//
//  Create the network derivatives of the cost function with respect to the biases
//  Return a pointer to the head of the linked list
//
void * _Nonnull initNetworkCostBiaseDerivatives(int * _Nonnull ntLayers, unsigned int numberOfLayers) {
    
    costBiaseDerivativeNode *dcdbList = (costBiaseDerivativeNode *)allocateCostBiaseDerivativeNode();
    
    // The first bias node (i.e., layer)
    dcdbList->dcdb = floatvec(0, ntLayers[1]-1);
    dcdbList->n = ntLayers[1];
    // The rest of the bias nodes (i.e., layers)
    int idx = 2;
    int k = 1;
    costBiaseDerivativeNode *dcdbNodePt = dcdbList;
    while (k < numberOfLayers-1) {
        costBiaseDerivativeNode *newNode = (costBiaseDerivativeNode *)allocateCostBiaseDerivativeNode();
        newNode->dcdb = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = dcdbNodePt;
        dcdbNodePt->next = newNode;
        dcdbNodePt = newNode;
        k++;
        idx++;
    }
    
    return (void *)dcdbList;
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
                parseArgument(field->value, field->key, nn->parameters->topology, &nn->parameters->numberOfLayers, &len);
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
                
                if (nn->parameters->numberOfActivationFunctions > 1 && nn->parameters->numberOfActivationFunctions < nn->parameters->numberOfLayers-1) {
                    fatal(DEFAULT_CONSOLE_WRITER, "the number of activation functions in parameters is too low. Can't resolve how to use the provided activations. ");
                }
                if (nn->parameters->numberOfActivationFunctions > nn->parameters->numberOfLayers-1) {
                    fprintf(stdout, "%s: too many activation functions given to network. Will ignore the extra ones.\n", DEFAULT_CONSOLE_WRITER);
                }
                
                if (nn->parameters->numberOfActivationFunctions == 1) {
                    if (strcmp(nn->parameters->activationFunctions[0], "softmax") == 0) {
                        fatal(DEFAULT_CONSOLE_WRITER, "the softmax function can only be used for the output units, not for the entire network.");
                    }
                    for (int i=0; i<nn->parameters->numberOfLayers-1; i++) {
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
                    for (int i=0; i<nn->parameters->numberOfLayers-1; i++) {
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
                            if (i < nn->parameters->numberOfLayers-2) {
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
                for (int i=0; i<nn->parameters->numberOfLayers-1; i++) {
                    nn->regularizer[i] = nn->l1_regularizer;
                    FOUND_REGULARIZATION = 1;
                }
            } else if (strcmp(field->key, "l2_regularization") == 0) {
                if (FOUND_TOPOLOGY == 0) {
                    fatal(DEFAULT_CONSOLE_WRITER, "incorrect parameters definition order, the topology is not defined yet. ");
                }
                nn->parameters->lambda = strtof(field->value, NULL);
                for (int i=0; i<nn->parameters->numberOfLayers-1; i++) {
                    nn->regularizer[i] = nn->l2_regularizer;
                    FOUND_REGULARIZATION = 1;
                }
            } else if (strcmp(field->key, "momentum") == 0) {
                nn->parameters->mu = strtof(field->value, NULL);
            
            } else if (strcmp(field->key, "adagrad") == 0) {
                nn->adaGrad = (AdaGrad *)malloc(sizeof(AdaGrad));
                nn->adaGrad->delta = strtof(field->value, NULL);
                nn->adaGrad->costWeightDerivativeSquaredAccumulated = NULL;
                nn->adaGrad->costBiasDerivativeSquaredAccumulated = NULL;
                nn->adapativeLearningRateMethod = ADAGRAD;
            
            } else if (strcmp(field->key, "rmsprop") == 0) {
                nn->rmsProp = (RMSProp *)malloc(sizeof(RMSProp));
                float result[2];
                unsigned int numberOfItems, len = 2;
                parseArgument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 2) fatal(DEFAULT_CONSOLE_WRITER, "the decay rate and a small constant should be given for the RMSProp method.");
                nn->rmsProp->decayRate = result[0];
                nn->rmsProp->delta = result[1];
                nn->rmsProp->costWeightDerivativeSquaredAccumulated = NULL;
                nn->rmsProp->costBiasDerivativeSquaredAccumulated = NULL;
                nn->adapativeLearningRateMethod = RMSPROP;
            
            } else if (strcmp(field->key, "adam") == 0) {
                nn->adam = (Adam *)malloc(sizeof(Adam));
                float result[4];
                unsigned int numberOfItems, len=4;
                parseArgument(field->value, field->key, result, &numberOfItems, &len);
                if (numberOfItems < 4) fatal(DEFAULT_CONSOLE_WRITER, "The step size, two decay rates and a small constant should be given for the Adam method.");
                nn->adam->time = 0;
                nn->adam->stepSize = result[0];
                nn->adam->decayRate1 = result[1];
                nn->adam->decayRate2 = result[2];
                nn->adam->delta = result[3];
                nn->adam->weightsBiasedFirstMomentEstimate = NULL;
                nn->adam->weightsBiasedSecondMomentEstimate = NULL;
                nn->adam->biasesBiasedFirstMomentEstimate = NULL;
                nn->adam->biasesBiasedSecondMomentEstimate = NULL;
                nn->adapativeLearningRateMethod = ADAM;
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
        for (int i=0; i<nn->parameters->numberOfLayers-1; i++) {
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
        for (int i=0; i<nn->parameters->numberOfLayers-1; i++) {
            nn->regularizer[i] = nn->l0_regularizer;
        }
    }
    
    
    return 0;
}
