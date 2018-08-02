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

static void genesis(void * _Nonnull self, char * _Nonnull init_stategy);
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
// Allocate memory for a neural network
//
NeuralNetwork * _Nonnull newNeuralNetwork(void) {
    
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    *nn = (NeuralNetwork){.dense_weights=NULL, .dense_weightsVelocity=NULL, .dense_biases=NULL, .dense_biasesVelocity=NULL, .dense_activations=NULL, .dense_affineTransformations=NULL, .dense_costWeightDerivatives=NULL, .dense_costBiasDerivatives=NULL, .dense_batchCostWeightDeriv=NULL, .dense_batchCostBiasDeriv=NULL};
    
    nn->parameters = (networkParameters *)malloc(sizeof(networkParameters));
    strcpy(nn->parameters->supported_parameters[0], "data_name");
    strcpy(nn->parameters->supported_parameters[1], "data");
    strcpy(nn->parameters->supported_parameters[2], "topology");
    strcpy(nn->parameters->supported_parameters[3], "activations");
    strcpy(nn->parameters->supported_parameters[4], "split");
    strcpy(nn->parameters->supported_parameters[5], "classification");
    strcpy(nn->parameters->supported_parameters[6], "epochs");
    strcpy(nn->parameters->supported_parameters[7], "batch_size");
    strcpy(nn->parameters->supported_parameters[8], "l1_regularization");
    strcpy(nn->parameters->supported_parameters[9], "l2_regularization");
    strcpy(nn->parameters->supported_parameters[10], "gradient_descent_optimizer");
    strcpy(nn->parameters->supported_parameters[11], "momentum_optimizer");
    strcpy(nn->parameters->supported_parameters[12], "adagrad_optimizer");
    strcpy(nn->parameters->supported_parameters[13], "rmsprop_optimizer");
    strcpy(nn->parameters->supported_parameters[14], "adam_optimizer");
    
    bzero(nn->parameters->data, MAX_LONG_STRING_LENGTH);
    strcpy(nn->parameters->dataName, "<empty>");
    nn->parameters->epochs = 0;
    nn->parameters->miniBatchSize = 0;
    nn->parameters->eta = 0.0f;
    nn->parameters->lambda = 0.0f;
    nn->network_num_layers = 0;
    nn->num_dense_layers = 0;
    nn->num_conv2d_layers = 0;
    nn->parameters->numberOfActivationFunctions = 0;
    nn->parameters->numberOfClassifications = 0;
    memset(nn->parameters->topology, 0, sizeof(nn->parameters->topology));
    memset(nn->parameters->classifications, 0, sizeof(nn->parameters->classifications));
    memset(nn->parameters->split, 0, sizeof(nn->parameters->split));
    
    memset(*nn->parameters->activationFunctions, 0, (MAX_NUMBER_NETWORK_LAYERS*128)*sizeof(char));
    
    for (int i=0; i<MAX_NUMBER_NETWORK_LAYERS; i++) {
        nn->activationFunctions[i] = NULL;
        nn->activationDerivatives[i] = NULL;
    }
    
    nn->constructor = allocateConstructor();
    nn->train = (Train *)malloc(sizeof(Train));
    nn->train->gradient_descent = NULL;
    nn->train->ada_grad = NULL;
    nn->train->rms_prop = NULL;
    nn->train->adam = NULL;
    nn->train->next_batch = nextBatch;
    nn->train->batch_range = batchRange;
    nn->train->progression = progression;
    
    nn->load_params_from_input_file = loadParametersFromImputFile;
    
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
//  Create the network layers, i.e. allocates memory for the weight, bias, activation, z, dC/dx and dC/db data structures
//
static void genesis(void * _Nonnull self, char * _Nonnull init_stategy) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // If the network is constructed with the constructor API, check that all required parameters were defined
    if (nn->constructor->networkConstruction) {
        if (nn->network_num_layers == 0) fatal(DEFAULT_CONSOLE_WRITER, "topology not defined. Use a constructor or define it in a parameter file.");
        
        if (nn->parameters->numberOfActivationFunctions == 0) {
            for (int i=0; i<nn->network_num_layers-1; i++) {
                nn->activationFunctions[i] = sigmoid;
                nn->activationDerivatives[i] = sigmoidPrime;
            }
        }
        
        if (nn->parameters->split[0] == 0 || nn->parameters->split[1] == 0) fatal(DEFAULT_CONSOLE_WRITER, "data split not defined. Use a constructor or define it in a parameter file.");
        
        if (nn->parameters->numberOfClassifications == 0) fatal(DEFAULT_CONSOLE_WRITER, "classification not defined. Use a constructor or define it in a parameter file.");
        
        char testString[MAX_LONG_STRING_LENGTH];
        bzero(testString, MAX_LONG_STRING_LENGTH);
        if (strcmp(nn->parameters->data, testString) == 0) fatal(DEFAULT_CONSOLE_WRITER, "training data not defined. Use a constructor or define it in a parameter file.");
    }
    
    fprintf(stdout, "%s: create the network internal structure...\n", DEFAULT_CONSOLE_WRITER);
    fprintf(stdout, "%s: full connected network with %d layers.\n", DEFAULT_CONSOLE_WRITER, nn->network_num_layers);
    
    nn->example_idx = 0;
    nn->parameters->number_of_features = nn->parameters->topology[0];;
    nn->parameters->max_number_of_nodes_in_layer = max_array(nn->parameters->topology, nn->network_num_layers);
    
    nn->data = (data *)malloc(sizeof(data));
    nn->data->init = initNeuralData;
    nn->data->load = loadData;
    
    if (nn->dense_weights == NULL) {
        tensor_dict dict;
        dict.rank = 2;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            dict.shape[l][0][0] = nn->parameters->topology[l+1];
            dict.shape[l][1][0] = nn->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = true;
        dict.init_strategy = init_stategy;
        nn->dense_weights = (tensor *)nn->tensor((void *)self, dict);
        
        if (nn->train->momentum != NULL) {
            if (nn->dense_weightsVelocity == NULL) {
                dict.init = false;
                nn->dense_weightsVelocity = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->train->ada_grad != NULL) {
            if (nn->train->ada_grad->costWeightDerivativeSquaredAccumulated == NULL) {
                dict.init = false;
                nn->train->ada_grad->costWeightDerivativeSquaredAccumulated = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->train->rms_prop != NULL) {
            if (nn->train->rms_prop->costWeightDerivativeSquaredAccumulated == NULL) {
                dict.init = false;
                nn->train->rms_prop->costWeightDerivativeSquaredAccumulated = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->train->adam != NULL) {
            dict.init = false;
            if (nn->train->adam->weightsBiasedFirstMomentEstimate == NULL) {
                nn->train->adam->weightsBiasedFirstMomentEstimate = (tensor *)nn->tensor((void *)self, dict);
            }
            if (nn->train->adam->weightsBiasedSecondMomentEstimate == NULL) {
                nn->train->adam->weightsBiasedSecondMomentEstimate = (tensor *)nn->tensor((void *)self,  dict);
            }
        }
    }
    
    if (nn->dense_biases == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=1; l<nn->network_num_layers; l++) {
            dict.shape[l-1][0][0] = nn->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = true;
        dict.init_strategy = init_stategy;
        nn->dense_biases = (tensor *)nn->tensor((void *)self, dict);
        
        if (nn->train->momentum != NULL) {
            if (nn->dense_biasesVelocity ==  NULL) {
                dict.init = false;
                nn->dense_biasesVelocity = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->train->ada_grad != NULL) {
            if (nn->train->ada_grad->costBiasDerivativeSquaredAccumulated == NULL) {
                dict.init = false;
                nn->train->ada_grad->costBiasDerivativeSquaredAccumulated = (tensor *)nn->tensor((void *)self, dict);
            }
        }
        if (nn->train->rms_prop != NULL) {
            if (nn->train->rms_prop->costBiasDerivativeSquaredAccumulated == NULL) {
                dict.init = false;
                nn->train->rms_prop->costBiasDerivativeSquaredAccumulated = (tensor *)nn->tensor((void *)self, dict);
            }
        }
         if (nn->train->adam != NULL) {
             dict.init = false;
             if (nn->train->adam->biasesBiasedFirstMomentEstimate == NULL) {
                 nn->train->adam->biasesBiasedFirstMomentEstimate = (tensor *)nn->tensor((void *)self, dict);
             }
             if (nn->train->adam->biasesBiasedSecondMomentEstimate == NULL) {
                 nn->train->adam->biasesBiasedSecondMomentEstimate = (tensor *)nn->tensor((void *)self, dict);
             }
         
         }
    }
    
    if (nn->dense_activations == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=0; l<nn->network_num_layers; l++) {
            dict.shape[l][0][0] = nn->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers;
        dict.init = false;
        nn->dense_activations = (tensor *)nn->tensor((void *)self, dict);
    }

    if (nn->dense_affineTransformations == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=1; l<nn->network_num_layers; l++) {
            dict.shape[l-1][0][0] = nn->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = false;
        nn->dense_affineTransformations = (tensor *)nn->tensor((void *)self, dict);
    }
    
    if (nn->dense_costWeightDerivatives == NULL) {
        tensor_dict dict;
        dict.rank = 2;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            dict.shape[l][0][0] = nn->parameters->topology[l+1];
            dict.shape[l][1][0] = nn->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = false;
        nn->dense_costWeightDerivatives = (tensor *)nn->tensor((void *)self, dict);
    }
    
    if (nn->dense_costBiasDerivatives == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=1; l<nn->network_num_layers; l++) {
            dict.shape[l-1][0][0] = nn->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = false;
        nn->dense_costBiasDerivatives = (tensor *)nn->tensor((void *)self, dict);
    }
    
    if (nn->dense_batchCostWeightDeriv == NULL) {
        tensor_dict dict;
        dict.rank = 2;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            dict.shape[l][0][0] = nn->parameters->topology[l+1];
            dict.shape[l][1][0] = nn->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = false;
        nn->dense_batchCostWeightDeriv = (tensor *)nn->tensor((void *)self, dict);
    }
    
    if (nn->dense_batchCostBiasDeriv == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=1; l<nn->network_num_layers; l++) {
            dict.shape[l-1][0][0] = nn->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init = false;
        nn->dense_batchCostBiasDeriv = (tensor *)nn->tensor((void *)self, dict);
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
    free(nn->parameters);
    free(nn->constructor);
    
    if (nn->dense_weights != NULL) {
        free(nn->dense_weights->val);
        free(nn->dense_weights);
    }
    
    if (nn->dense_biases != NULL) {
        free(nn->dense_biases->val);
        free(nn->dense_biases);
    }
    
    if (nn->dense_costWeightDerivatives != NULL) {
        free(nn->dense_costWeightDerivatives->val);
        free(nn->dense_costWeightDerivatives);
    }
    
    if (nn->dense_batchCostWeightDeriv != NULL) {
        free(nn->dense_batchCostWeightDeriv->val);
        free(nn->dense_batchCostWeightDeriv);
    }
    
    if (nn->dense_costBiasDerivatives != NULL) {
        free(nn->dense_costBiasDerivatives->val);
        free(nn->dense_costBiasDerivatives);
    }
    
    if (nn->dense_batchCostBiasDeriv != NULL) {
        free(nn->dense_batchCostBiasDeriv->val);
        free(nn->dense_batchCostBiasDeriv);
    }
    
    if (nn->dense_activations != NULL) {
        free(nn->dense_activations->val);
        free(nn->dense_activations);
    }
    
    if (nn->dense_affineTransformations != NULL) {
        free(nn->dense_affineTransformations->val);
        free(nn->dense_affineTransformations);
    }
    
    if (nn->gpu != NULL) {
        nn->gpu->nullify();
        free(nn->gpu);
    }
    
    if (nn->train->gradient_descent != NULL) {
        free(nn->train->gradient_descent);
    }
    if (nn->train->momentum != NULL) {
        if (nn->dense_weightsVelocity != NULL) {
            free(nn->dense_weightsVelocity->val);
            free(nn->dense_weightsVelocity);
        }
        if (nn->dense_biasesVelocity != NULL) {
            free(nn->dense_biasesVelocity->val);
            free(nn->dense_biasesVelocity);
        }
        free(nn->train->momentum);
    }
    if (nn->train->ada_grad != NULL) {
        if (nn->train->ada_grad->costWeightDerivativeSquaredAccumulated != NULL) {
            free(nn->train->ada_grad->costWeightDerivativeSquaredAccumulated->val);
            free(nn->train->ada_grad->costWeightDerivativeSquaredAccumulated);
        }
        if (nn->train->ada_grad->costBiasDerivativeSquaredAccumulated != NULL) {
            free(nn->train->ada_grad->costBiasDerivativeSquaredAccumulated->val);
            free(nn->train->ada_grad->costBiasDerivativeSquaredAccumulated);
        }
        free(nn->train->ada_grad);
    }
    if (nn->train->rms_prop != NULL) {
        if (nn->train->rms_prop->costWeightDerivativeSquaredAccumulated != NULL) {
            free(nn->train->rms_prop->costWeightDerivativeSquaredAccumulated->val);
            free(nn->train->rms_prop->costWeightDerivativeSquaredAccumulated);
        }
        if (nn->train->rms_prop->costBiasDerivativeSquaredAccumulated != NULL) {
            free(nn->train->rms_prop->costBiasDerivativeSquaredAccumulated->val);
            free(nn->train->rms_prop->costBiasDerivativeSquaredAccumulated);
        }
        free(nn->train->rms_prop);
    }
    if (nn->train->adam != NULL) {
        if (nn->train->adam->weightsBiasedFirstMomentEstimate != NULL) {
            free(nn->train->adam->weightsBiasedFirstMomentEstimate->val);
            free(nn->train->adam->weightsBiasedFirstMomentEstimate);
        }
        if (nn->train->adam->weightsBiasedSecondMomentEstimate != NULL) {
            free(nn->train->adam->weightsBiasedSecondMomentEstimate->val);
            free(nn->train->adam->weightsBiasedSecondMomentEstimate);
        }
        
        if (nn->train->adam->biasesBiasedFirstMomentEstimate != NULL) {
            free(nn->train->adam->biasesBiasedFirstMomentEstimate->val);
            free(nn->train->adam->biasesBiasedFirstMomentEstimate);
        }
        if (nn->train->adam->biasesBiasedSecondMomentEstimate != NULL) {
            free(nn->train->adam->biasesBiasedSecondMomentEstimate->val);
            free(nn->train->adam->biasesBiasedSecondMomentEstimate);
        }
        free(nn->train->adam);
    }
    free(nn->train);
}

static void gpu_alloc(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->gpu = metalCompute();
}
