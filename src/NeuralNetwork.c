//
//  NeuralNetwork.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 31/05/2017.
//

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include "cblas.h"
    #include "cblas_f77.h"
#endif

#include "NeuralNetwork.h"
#include "Memory.h"
#include "Parsing.h"
#include "Regularization.h"
#include "TimeProfile.h"
#include "NetworkOps.h"

static void initNeuralData(void * _Nonnull self);

static void genesis(void * _Nonnull self);
static void finale(void * _Nonnull self);
static void gpu_alloc(void * _Nonnull self);

static void initNeuralData(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->data->training = (training *)malloc(sizeof(training));
    nn->data->training->set = NULL;
    nn->data->training->m = 0;
    nn->data->training->n = 0;
    
    nn->data->test = (test *)malloc(sizeof(test));
    nn->data->test->set = NULL;
    nn->data->test->m = 0;
    nn->data->test->n = 0;
    
    nn->data->validation = (validation *)malloc(sizeof(validation));
    nn->data->validation->set = NULL;
    nn->data->validation->m = 0;
    nn->data->validation->n = 0;
}

//
// Full-connected network allocation
//
static void dense_net_create(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->dense = (dense_network *)malloc(sizeof(dense_network));
    *(nn->dense) = (dense_network){.num_dense_layers=0, .weights=NULL, .weightsVelocity=NULL, .biases=NULL, .biasesVelocity=NULL, .activations=NULL, .affineTransformations=NULL, .costWeightDerivatives=NULL, .costBiasDerivatives=NULL, .batchCostWeightDeriv=NULL, .batchCostBiasDeriv=NULL};
    nn->dense->train = (Train *)malloc(sizeof(Train));
    *(nn->dense->train) = (Train){.gradient_descent=NULL, .ada_grad=NULL, .rms_prop=NULL,. adam=NULL};
    nn->dense->train->next_batch = nextBatch;
    nn->dense->train->batch_range = batchRange;
    nn->dense->train->progression = progression;
    
    for (int i=0; i<MAX_NUMBER_NETWORK_LAYERS; i++) {
        nn->dense->activationFunctions[i] = NULL;
        nn->dense->activationDerivatives[i] = NULL;
    }
    
    nn->dense->parameters = (dense_net_parameters *)malloc(sizeof(dense_net_parameters));
    strcpy(nn->dense->parameters->supported_parameters[0], "data_name");
    strcpy(nn->dense->parameters->supported_parameters[1], "data");
    strcpy(nn->dense->parameters->supported_parameters[2], "topology");
    strcpy(nn->dense->parameters->supported_parameters[3], "activations");
    strcpy(nn->dense->parameters->supported_parameters[4], "split");
    strcpy(nn->dense->parameters->supported_parameters[5], "classification");
    strcpy(nn->dense->parameters->supported_parameters[6], "epochs");
    strcpy(nn->dense->parameters->supported_parameters[7], "batch_size");
    strcpy(nn->dense->parameters->supported_parameters[8], "l1_regularization");
    strcpy(nn->dense->parameters->supported_parameters[9], "l2_regularization");
    strcpy(nn->dense->parameters->supported_parameters[10], "gradient_descent_optimizer");
    strcpy(nn->dense->parameters->supported_parameters[11], "momentum_optimizer");
    strcpy(nn->dense->parameters->supported_parameters[12], "adagrad_optimizer");
    strcpy(nn->dense->parameters->supported_parameters[13], "rmsprop_optimizer");
    strcpy(nn->dense->parameters->supported_parameters[14], "adam_optimizer");
    
    nn->dense->parameters->epochs = 0;
    nn->dense->parameters->miniBatchSize = 0;
    nn->dense->parameters->eta = 0.0f;
    nn->dense->parameters->lambda = 0.0f;
    nn->dense->parameters->numberOfClassifications = 0;
    memset(nn->dense->parameters->topology, 0, sizeof(nn->dense->parameters->topology));
    memset(nn->dense->parameters->classifications, 0, sizeof(nn->dense->parameters->classifications));
    memset(nn->dense->parameters->split, 0, sizeof(nn->dense->parameters->split));
    nn->dense->load_params_from_input_file = loadParametersFromImputFile;
}

//
// Full-connected genesis
//
static void dense_net_genesis(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->dense->parameters->max_number_of_nodes_in_layer = max_array(nn->dense->parameters->topology, nn->network_num_layers);
    
    if (nn->dense->parameters->split[0] == 0 || nn->dense->parameters->split[1] == 0) fatal(DEFAULT_CONSOLE_WRITER, "data split not defined. Use a constructor or define it in a parameter file.");
    
    if (nn->dense->parameters->numberOfClassifications == 0) fatal(DEFAULT_CONSOLE_WRITER, "classification not defined. Use a constructor or define it in a parameter file.");
    
    if (nn->dense->weights == NULL) {
        tensor_dict dict;
        dict.rank = 2;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            dict.shape[l][0][0] = nn->dense->parameters->topology[l+1];
            dict.shape[l][1][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = true;
        nn->dense->weights = (tensor *)nn->tensor((void *)self, dict);
        
        if (nn->dense->train->momentum != NULL) {
            if (nn->dense->weightsVelocity == NULL) {
                dict.init = false;
                nn->dense->weightsVelocity = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->dense->train->ada_grad != NULL) {
            if (nn->dense->train->ada_grad->costWeightDerivativeSquaredAccumulated == NULL) {
                dict.init = false;
                nn->dense->train->ada_grad->costWeightDerivativeSquaredAccumulated = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->dense->train->rms_prop != NULL) {
            if (nn->dense->train->rms_prop->costWeightDerivativeSquaredAccumulated == NULL) {
                dict.init = false;
                nn->dense->train->rms_prop->costWeightDerivativeSquaredAccumulated = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->dense->train->adam != NULL) {
            dict.init = false;
            if (nn->dense->train->adam->weightsBiasedFirstMomentEstimate == NULL) {
                nn->dense->train->adam->weightsBiasedFirstMomentEstimate = (tensor *)nn->tensor((void *)self, dict);
            }
            if (nn->dense->train->adam->weightsBiasedSecondMomentEstimate == NULL) {
                nn->dense->train->adam->weightsBiasedSecondMomentEstimate = (tensor *)nn->tensor((void *)self,  dict);
            }
        }
    }
    
    if (nn->dense->biases == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=1; l<nn->network_num_layers; l++) {
            dict.shape[l-1][0][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = true;
        nn->dense->biases = (tensor *)nn->tensor((void *)self, dict);
        
        if (nn->dense->train->momentum != NULL) {
            if (nn->dense->biasesVelocity ==  NULL) {
                dict.init = false;
                nn->dense->biasesVelocity = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->dense->train->ada_grad != NULL) {
            if (nn->dense->train->ada_grad->costBiasDerivativeSquaredAccumulated == NULL) {
                dict.init = false;
                nn->dense->train->ada_grad->costBiasDerivativeSquaredAccumulated = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->dense->train->rms_prop != NULL) {
            if (nn->dense->train->rms_prop->costBiasDerivativeSquaredAccumulated == NULL) {
                dict.init = false;
                nn->dense->train->rms_prop->costBiasDerivativeSquaredAccumulated = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->dense->train->adam != NULL) {
            dict.init = false;
            if (nn->dense->train->adam->biasesBiasedFirstMomentEstimate == NULL) {
                nn->dense->train->adam->biasesBiasedFirstMomentEstimate = (tensor *)nn->tensor((void *)self, dict);
            }
            if (nn->dense->train->adam->biasesBiasedSecondMomentEstimate == NULL) {
                nn->dense->train->adam->biasesBiasedSecondMomentEstimate = (tensor *)nn->tensor((void *)self, dict);
            }
        }
    }
    
    if (nn->dense->activations == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=0; l<nn->network_num_layers; l++) {
            dict.shape[l][0][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers;
        dict.init = false;
        nn->dense->activations = (tensor *)nn->tensor((void *)self, dict);
    }
    
    if (nn->dense->affineTransformations == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=1; l<nn->network_num_layers; l++) {
            dict.shape[l-1][0][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = false;
        nn->dense->affineTransformations = (tensor *)nn->tensor((void *)self, dict);
    }
    
    if (nn->dense->costWeightDerivatives == NULL) {
        tensor_dict dict;
        dict.rank = 2;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            dict.shape[l][0][0] = nn->dense->parameters->topology[l+1];
            dict.shape[l][1][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = false;
        nn->dense->costWeightDerivatives = (tensor *)nn->tensor((void *)self, dict);
    }
    
    if (nn->dense->costBiasDerivatives == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=1; l<nn->network_num_layers; l++) {
            dict.shape[l-1][0][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = false;
        nn->dense->costBiasDerivatives = (tensor *)nn->tensor((void *)self, dict);
    }
    
    if (nn->dense->batchCostWeightDeriv == NULL) {
        tensor_dict dict;
        dict.rank = 2;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            dict.shape[l][0][0] = nn->dense->parameters->topology[l+1];
            dict.shape[l][1][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = false;
        nn->dense->batchCostWeightDeriv = (tensor *)nn->tensor((void *)self, dict);
    }
    
    if (nn->dense->batchCostBiasDeriv == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=1; l<nn->network_num_layers; l++) {
            dict.shape[l-1][0][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = false;
        nn->dense->batchCostBiasDeriv = (tensor *)nn->tensor((void *)self, dict);
    }
}


static void dense_net_finale(void * _Nonnull  self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    if (nn->dense->weights != NULL) {
        free(nn->dense->weights->val);
        free(nn->dense->weights);
    }
    if (nn->dense->train->momentum != NULL) {
        if (nn->dense->weightsVelocity != NULL) {
            free(nn->dense->weightsVelocity->val);
            free(nn->dense->weightsVelocity);
        }
        if (nn->dense->biasesVelocity != NULL) {
            free(nn->dense->biasesVelocity->val);
            free(nn->dense->biasesVelocity);
        }
        free(nn->dense->train->momentum);
    }
    if (nn->dense->biases != NULL) {
        free(nn->dense->biases->val);
        free(nn->dense->biases);
    }
    if (nn->dense->activations != NULL) {
        free(nn->dense->activations->val);
        free(nn->dense->activations);
    }
    if (nn->dense->affineTransformations != NULL) {
        free(nn->dense->affineTransformations->val);
        free(nn->dense->affineTransformations);
    }
    if (nn->dense->costWeightDerivatives != NULL) {
        free(nn->dense->costWeightDerivatives->val);
        free(nn->dense->costWeightDerivatives);
    }
    if (nn->dense->costBiasDerivatives != NULL) {
        free(nn->dense->costBiasDerivatives->val);
        free(nn->dense->costBiasDerivatives);
    }
    if (nn->dense->batchCostWeightDeriv != NULL) {
        free(nn->dense->batchCostWeightDeriv->val);
        free(nn->dense->batchCostWeightDeriv);
    }
    if (nn->dense->batchCostBiasDeriv != NULL) {
        free(nn->dense->batchCostBiasDeriv->val);
        free(nn->dense->batchCostBiasDeriv);
    }
    if (nn->dense->train->gradient_descent != NULL) {
        free(nn->dense->train->gradient_descent);
    }
    
    if (nn->dense->train->ada_grad != NULL) {
        if (nn->dense->train->ada_grad->costWeightDerivativeSquaredAccumulated != NULL) {
            free(nn->dense->train->ada_grad->costWeightDerivativeSquaredAccumulated->val);
            free(nn->dense->train->ada_grad->costWeightDerivativeSquaredAccumulated);
        }
        if (nn->dense->train->ada_grad->costBiasDerivativeSquaredAccumulated != NULL) {
            free(nn->dense->train->ada_grad->costBiasDerivativeSquaredAccumulated->val);
            free(nn->dense->train->ada_grad->costBiasDerivativeSquaredAccumulated);
        }
        free(nn->dense->train->ada_grad);
    }
    if (nn->dense->train->rms_prop != NULL) {
        if (nn->dense->train->rms_prop->costWeightDerivativeSquaredAccumulated != NULL) {
            free(nn->dense->train->rms_prop->costWeightDerivativeSquaredAccumulated->val);
            free(nn->dense->train->rms_prop->costWeightDerivativeSquaredAccumulated);
        }
        if (nn->dense->train->rms_prop->costBiasDerivativeSquaredAccumulated != NULL) {
            free(nn->dense->train->rms_prop->costBiasDerivativeSquaredAccumulated->val);
            free(nn->dense->train->rms_prop->costBiasDerivativeSquaredAccumulated);
        }
        free(nn->dense->train->rms_prop);
    }
    if (nn->dense->train->adam != NULL) {
        if (nn->dense->train->adam->weightsBiasedFirstMomentEstimate != NULL) {
            free(nn->dense->train->adam->weightsBiasedFirstMomentEstimate->val);
            free(nn->dense->train->adam->weightsBiasedFirstMomentEstimate);
        }
        if (nn->dense->train->adam->weightsBiasedSecondMomentEstimate != NULL) {
            free(nn->dense->train->adam->weightsBiasedSecondMomentEstimate->val);
            free(nn->dense->train->adam->weightsBiasedSecondMomentEstimate);
        }
        
        if (nn->dense->train->adam->biasesBiasedFirstMomentEstimate != NULL) {
            free(nn->dense->train->adam->biasesBiasedFirstMomentEstimate->val);
            free(nn->dense->train->adam->biasesBiasedFirstMomentEstimate);
        }
        if (nn->dense->train->adam->biasesBiasedSecondMomentEstimate != NULL) {
            free(nn->dense->train->adam->biasesBiasedSecondMomentEstimate->val);
            free(nn->dense->train->adam->biasesBiasedSecondMomentEstimate);
        }
        free(nn->dense->train->adam);
    }
    free(nn->dense->train);
    free(nn->dense);
}

//
// Allocate memory for a neural network
//
NeuralNetwork * _Nonnull newNeuralNetwork(void) {
    
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    *nn = (NeuralNetwork){.network_num_layers=0, .is_dense_network=true, .is_conv2d_network=false};
    
    dense_net_create((void *)nn);
    
    nn->num_activation_functions = 0;
    memset(*nn->activationFunctionsStr, 0, (MAX_NUMBER_NETWORK_LAYERS*128)*sizeof(char));
    
    bzero(nn->dataPath, MAX_LONG_STRING_LENGTH);
    strcpy(nn->dataName, "<empty>");
    
    nn->constructor = allocateConstructor();
    
    nn->genesis = genesis;
    nn->finale = finale;
    nn->tensor = tensor_create;
    nn->gpu_alloc = gpu_alloc;
    
    nn->l0_regularizer = l0_regularizer;
    nn->l1_regularizer = l1_regularizer;
    nn->l2_regularizer = l2_regularizer;
    
    // This function is only used when loading a network from a param file
    nn->train_loop = trainLoop;
    
    nn->math_ops = mathOps;
    
    nn->eval_prediction = evalPrediction;
    nn->eval_cost = evalCost;
    
    return nn;
}

//
//  The network genesis
//
static void genesis(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // If the network is constructed with the constructor API, check that all required parameters were defined
    if (nn->constructor->networkConstruction) {
        if (nn->network_num_layers == 0) fatal(DEFAULT_CONSOLE_WRITER, "topology not defined. Use a constructor or define it in a parameter file.");
        
        if (nn->num_activation_functions == 0) {
            for (int i=0; i<nn->network_num_layers-1; i++) {
                nn->dense->activationFunctions[i] = sigmoid;
                nn->dense->activationDerivatives[i] = sigmoidPrime;
            }
        }
        
        
        char testString[MAX_LONG_STRING_LENGTH];
        bzero(testString, MAX_LONG_STRING_LENGTH);
        if (strcmp(nn->dataPath, testString) == 0) fatal(DEFAULT_CONSOLE_WRITER, "training data not defined. Use a constructor or define it in a parameter file.");
    }
    
    fprintf(stdout, "%s: create the network internal structure...\n", DEFAULT_CONSOLE_WRITER);
    fprintf(stdout, "%s: full connected network with %d layers.\n", DEFAULT_CONSOLE_WRITER, nn->network_num_layers);
    
    nn->example_idx = 0;
    
    nn->data = (data *)malloc(sizeof(data));
    nn->data->init = initNeuralData;
    nn->data->load = loadData;
    
    if (nn->is_dense_network) {
        dense_net_genesis(self);
    } else if (nn->is_conv2d_network) {
        
    }
}

//
// Free-up all the memory used by the network
//
static void finale(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    free_fmatrix(nn->data->training->set, 0, nn->data->training->m, 0, nn->data->training->n);
    free_fmatrix(nn->data->test->set, 0, nn->data->test->m, 0, nn->data->test->n);
    if (nn->data->validation->set != NULL) free_fmatrix(nn->data->validation->set, 0, nn->data->validation->m, 0, nn->data->validation->n);
    nn->data->training->reader = NULL;
    nn->data->test->reader = NULL;
    free(nn->data->training);
    free(nn->data->test);
    free(nn->data->validation);
    nn->data->init = NULL;
    nn->data->load = NULL;
    free(nn->data);
    free(nn->dense->parameters);
    free(nn->constructor);
    
    if (nn->is_dense_network) {
        dense_net_finale(self);
    } else if (nn->is_conv2d_network) {
        
    }
    
    if (nn->gpu != NULL) {
        nn->gpu->nullify();
        free(nn->gpu);
    }
}

static void gpu_alloc(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->gpu = metalCompute();
}
