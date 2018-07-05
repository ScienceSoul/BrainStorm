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
//  Create the matrices of the network, typically used to create the weights and the weights velocity.
//  They are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1 over the square root of the number of
//  weights/velocities connecting to the same neuron.
//
float * _Nonnull initMatrices(int * _Nonnull topology, unsigned int numberOfLayers, bool gaussian) {
    
    int dim = 0;
    for (int l=0; l<numberOfLayers-1; l++) {
        dim = dim + (topology[l+1]*topology[l]);
    }
    fprintf(stdout, "%s: matrices allocation: allocate %f (MB)\n", PROGRAM_NAME, ((float)dim*sizeof(float))/(float)(1024*1024));
    float *matrices = (float *)malloc(dim*sizeof(float));
    
    if (gaussian) {
        int stride = 0;
        for (int l=0; l<numberOfLayers-1; l++) {
            int m = topology[l+1];
            int n = topology[l];
            for (int i = 0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    matrices[stride+((i*n)+j)] = randn(0.0f, 1.0f) / sqrtf((float)n);
                }
            }
            stride = stride + (m * n);
        }
    } else {
        memset(matrices, 0.0f, dim*sizeof(float));
    }
    
    return matrices;
}

//
//  Create the vectors of the network, typically used to create the biases and the biases velocity.
//  They are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1.
//
float * _Nonnull initVectors(int * _Nonnull topology, unsigned int numberOfLayers, bool gaussian) {
    
    int dim = 0;
    for (int l=1; l<numberOfLayers; l++) {
        dim  = dim + topology[l];
    }
    fprintf(stdout, "%s: vectors allocation: allocate %f (MB)\n", PROGRAM_NAME, ((float)dim*sizeof(float))/(float)(1024*1024));
    float *vectors = (float*)malloc(dim*sizeof(float));
    
    if (gaussian) {
        int stride = 0;
        for (int l=1; l<numberOfLayers; l++) {
            int n = topology[l];
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

int loadParameters(void * _Nonnull self, const char * _Nonnull paraFile) {
    
    definition *definitions = NULL;
    
    definitions = getDefinitions(self, paraFile, "define");
    if (definitions == NULL) {
        fatal(PROGRAM_NAME, "problem finding any parameter definition.");
    }
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    short FOUND_DATA_NAME      = 0;
    short FOUND_DATA           = 0;
    short FOUND_TOPOLOGY       = 0;
    short FOUND_ACTIVATIONS    = 0;
    short FOUND_SPLIT          = 0;
    short FOUND_CLASSIFICATION = 0;
    
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
            if (!found) fatal(PROGRAM_NAME, "key for parameter not recognized:", field->key);
            
            if (strcmp(field->key, "data_name") == 0) {
                strcpy(nn->parameters->dataName, field->value);
                FOUND_DATA_NAME = 1;
                
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
                    fatal(PROGRAM_NAME, "missing topology in parameters input.");
                }
                
                unsigned int len = MAX_NUMBER_NETWORK_LAYERS;
                parseArgument(field->value, field->key, nn->parameters->activationFunctions, &nn->parameters->numberOfActivationFunctions, &len);
                
                if (nn->parameters->numberOfActivationFunctions > 1 && nn->parameters->numberOfActivationFunctions < nn->parameters->numberOfLayers-1) {
                    fatal(PROGRAM_NAME, "the number of activation functions in parameters is too low. Can't resolve how to use the provided activations. ");
                }
                if (nn->parameters->numberOfActivationFunctions > nn->parameters->numberOfLayers-1) {
                    fprintf(stdout, "%s: too many activation functions given to network. Will ignore the extra ones.\n", PROGRAM_NAME);
                }
                
                if (nn->parameters->numberOfActivationFunctions == 1) {
                    if (strcmp(nn->parameters->activationFunctions[0], "softmax") == 0) {
                        fatal(PROGRAM_NAME, "the softmax function can only be used for the output units and can't be used for the entire network.");
                    }
                    for (int i=0; i<nn->parameters->numberOfLayers-1; i++) {
                        if (strcmp(nn->parameters->activationFunctions[0], "sigmoid") == 0) {
                            nn->activationFunctions[i] = sigmoid;
                            nn->activationDerivatives[i] = sigmoidPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[0], "relu") == 0) {
                            nn->activationFunctions[i] = relu;
                            nn->activationDerivatives[i] = reluPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[0], "tanh") == 0) {
                            nn->activationFunctions[i] = tan_h;
                            nn->activationDerivatives[i] = tanhPrime;
                        } else fatal(PROGRAM_NAME, "unsupported or unrecognized activation function:", nn->parameters->activationFunctions[0]);
                    }
                } else {
                    for (int i=0; i<nn->parameters->numberOfLayers-1; i++) {
                        if (strcmp(nn->parameters->activationFunctions[i], "sigmoid") == 0) {
                            nn->activationFunctions[i] = sigmoid;
                            nn->activationDerivatives[i] = sigmoidPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[i], "relu") == 0) {
                            nn->activationFunctions[i] = relu;
                            nn->activationDerivatives[i] = reluPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[i], "tanh") == 0) {
                            nn->activationFunctions[i] = tan_h;
                            nn->activationDerivatives[i] = tanhPrime;
                        } else if (strcmp(nn->parameters->activationFunctions[i], "softmax") == 0) {
                            // The sofmax function is only supported for the output units
                            if (i < nn->parameters->numberOfLayers-2) {
                                fatal(PROGRAM_NAME, "the softmax function can't be used for the hiden units, only for the output units.");
                            }
                            nn->activationFunctions[i] = softmax;
                            nn->activationDerivatives[i] = NULL;
                        } else fatal(PROGRAM_NAME, "unsupported or unrecognized activation function:", nn->parameters->activationFunctions[i]);
                    }
                }
                FOUND_ACTIVATIONS = 1;
                
            } else if (strcmp(field->key, "split") == 0) {
                unsigned int n;
                unsigned int len = 2;
                parseArgument(field->value,  field->key, nn->parameters->split, &n, &len);
                if (n < 2) {
                    fatal(PROGRAM_NAME, " data splitting requires two values: one for training, one for testing/evaluation.");
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
                
            } else if (strcmp(field->key, "regularization_factor") == 0) {
                nn->parameters->lambda = strtof(field->value, NULL);
            
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
                if (numberOfItems < 2) fatal(PROGRAM_NAME, "the decay rate and a small constant should be given for the RMSProp method.");
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
                if (numberOfItems < 4) fatal(PROGRAM_NAME, "The step size, two decay rates and a small constant should be given for the Adam method.");
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
    
    if (FOUND_DATA_NAME == 0) {
        fatal(PROGRAM_NAME, "missing data name in parameters input.");
    }
    if (FOUND_DATA == 0) {
        fatal(PROGRAM_NAME, "missing data in parameters input.");
    }
    if (FOUND_TOPOLOGY == 0) {
        fatal(PROGRAM_NAME, "missing topology in parameters input.");
    }
    if (FOUND_ACTIVATIONS == 0) {
        for (int i=0; i<nn->parameters->numberOfLayers-1; i++) {
            nn->activationFunctions[i] = sigmoid;
            nn->activationDerivatives[i] = sigmoidPrime;
        }
    }
    if (FOUND_SPLIT == 0) {
        fatal(PROGRAM_NAME, "missing split in parameters input.");
    }
    if (FOUND_CLASSIFICATION == 0) {
        fatal(PROGRAM_NAME, "missing classification in parameters input.");
    }
    
    return 0;
}
