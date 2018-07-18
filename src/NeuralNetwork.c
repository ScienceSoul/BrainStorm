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

static void initNeuralData(void * _Nonnull self);

static void genesis(void * _Nonnull self, char * _Nonnull init_stategy);
static void finale(void * _Nonnull self);
static void gpu_alloc(void * _Nonnull self);

static int evaluate(void * _Nonnull self, bool metal);

static float totalCost(void * _Nonnull self, float * _Nonnull * _Nonnull data, unsigned int m, bool convert);

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
    *nn = (NeuralNetwork){.weights=NULL, .weightsVelocity=NULL, .biases=NULL, .biasesVelocity=NULL, .networkActivations=NULL, .networkAffineTransformations=NULL, .networkCostWeightDerivatives=NULL, .networkCostBiaseDerivatives=NULL, .deltaNetworkCostWeightDerivatives=NULL, .deltaNetworkCostBiaseDerivatives=NULL};
    
    nn->parameters = (networkParameters *)malloc(sizeof(networkParameters));
    strcpy(nn->parameters->supported_parameters[0], "data_name");
    strcpy(nn->parameters->supported_parameters[1], "data");
    strcpy(nn->parameters->supported_parameters[2], "topology");
    strcpy(nn->parameters->supported_parameters[3], "activations");
    strcpy(nn->parameters->supported_parameters[4], "split");
    strcpy(nn->parameters->supported_parameters[5], "classification");
    strcpy(nn->parameters->supported_parameters[6], "epochs");
    strcpy(nn->parameters->supported_parameters[7], "batch_size");
    strcpy(nn->parameters->supported_parameters[8], "learning_rate");
    strcpy(nn->parameters->supported_parameters[9], "l1_regularization");
    strcpy(nn->parameters->supported_parameters[10], "l2_regularization");
    strcpy(nn->parameters->supported_parameters[11], "gradient_descent_optimizer");
    strcpy(nn->parameters->supported_parameters[12], "momentum_optimizer");
    strcpy(nn->parameters->supported_parameters[13], "adagrad_optimizer");
    strcpy(nn->parameters->supported_parameters[14], "rmsprop_optimizer");
    strcpy(nn->parameters->supported_parameters[15], "adam_optimizer");
    
    bzero(nn->parameters->data, MAX_LONG_STRING_LENGTH);
    strcpy(nn->parameters->dataName, "<empty>");
    nn->parameters->epochs = 0;
    nn->parameters->miniBatchSize = 0;
    nn->parameters->eta = 0.0f;
    nn->parameters->lambda = 0.0f;
    nn->parameters->mu = 0.0f;
    nn->parameters->numberOfLayers = 0;
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
    nn->tensor = tensor;
    nn->gpu_alloc = gpu_alloc;
    nn->evaluate = evaluate;
    nn->totalCost = totalCost;
    
    nn->l0_regularizer = l0_regularizer;
    nn->l1_regularizer = l1_regularizer;
    nn->l2_regularizer = l2_regularizer;
    
    // This function is only used when loading a network from a param file
    nn->train_loop = trainLoop;
    
    return nn;
}

//
//  Create the network layers, i.e. allocates memory for the weight, bias, activation, z, dC/dx and dC/db data structures
//
static void genesis(void * _Nonnull self, char * _Nonnull init_stategy) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // If the network is constructed with the constructor API, check that all required parameters were defined
    if (nn->constructor->networkConstruction) {
        if (nn->parameters->numberOfLayers == 0) fatal(DEFAULT_CONSOLE_WRITER, "topology not defined. Use a constructor or define it in a parameter file.");
        
        if (nn->parameters->numberOfActivationFunctions == 0) {
            for (int i=0; i<nn->parameters->numberOfLayers-1; i++) {
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
    fprintf(stdout, "%s: full connected network with %d layers.\n", DEFAULT_CONSOLE_WRITER, nn->parameters->numberOfLayers);
    
    nn->example_idx = 0;
    nn->number_of_parameters = 0;
    nn->number_of_features = nn->parameters->topology[0];;
    nn->max_number_of_nodes_in_layer = max_array(nn->parameters->topology, nn->parameters->numberOfLayers);
    
    nn->data = (data *)malloc(sizeof(data));
    nn->data->init = initNeuralData;
    nn->data->load = loadData;
    
    for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
        nn->weightsDimensions[l].m = nn->parameters->topology[l+1];
        nn->weightsDimensions[l].n = nn->parameters->topology[l];
    }
    for (int l=1; l<nn->parameters->numberOfLayers; l++) {
        nn->biasesDimensions[l-1].n = nn->parameters->topology[l];
    }
    
    if (nn->weights == NULL)
        nn->weights = nn->tensor((void *)self, (tensor_dict){.rank=2, .init=true, .init_stategy=init_stategy});
    if (nn->biases == NULL)
        nn->biases = nn->tensor((void *)self, (tensor_dict){.rank=1, .init=true, .init_stategy=init_stategy});
    
    if (nn->networkActivations == NULL)
        nn->networkActivations = (activationNode *)initNetworkActivations(nn->parameters->topology, nn->parameters->numberOfLayers);
    if (nn->networkAffineTransformations == NULL)
        nn->networkAffineTransformations = (affineTransformationNode *)initNetworkAffineTransformations(nn->parameters->topology, nn->parameters->numberOfLayers);
    if (nn->networkCostWeightDerivatives == NULL)
        nn->networkCostWeightDerivatives = (costWeightDerivativeNode *)initNetworkCostWeightDerivatives(nn->parameters->topology, nn->parameters->numberOfLayers);
    if (nn->networkCostBiaseDerivatives == NULL)
        nn->networkCostBiaseDerivatives = (costBiaseDerivativeNode *)initNetworkCostBiaseDerivatives(nn->parameters->topology, nn->parameters->numberOfLayers);
    if (nn->deltaNetworkCostWeightDerivatives == NULL)
        nn->deltaNetworkCostWeightDerivatives = (costWeightDerivativeNode *)initNetworkCostWeightDerivatives(nn->parameters->topology, nn->parameters->numberOfLayers);
    if (nn->deltaNetworkCostBiaseDerivatives == NULL)
        nn->deltaNetworkCostBiaseDerivatives = (costBiaseDerivativeNode *)initNetworkCostBiaseDerivatives(nn->parameters->topology, nn->parameters->numberOfLayers);
    
    if (nn->train->momentum != NULL) {
        if (nn->weightsVelocity == NULL)
            nn->weightsVelocity = nn->tensor((void *)self, (tensor_dict){.rank=2, .init=false});
        if (nn->biasesVelocity == NULL)
            nn->biasesVelocity = nn->tensor((void *)self, (tensor_dict){.rank=1, .init=false});
    }
    
    if (nn->train->ada_grad != NULL) {
        if (nn->train->ada_grad->costWeightDerivativeSquaredAccumulated == NULL)
            nn->train->ada_grad->costWeightDerivativeSquaredAccumulated = nn->tensor((void *)self, (tensor_dict){.rank=2, .init=false});
        if (nn->train->ada_grad->costBiasDerivativeSquaredAccumulated == NULL)
            nn->train->ada_grad->costBiasDerivativeSquaredAccumulated = nn->tensor((void *)self, (tensor_dict){.rank=1, .init=false});
    }
    if (nn->train->rms_prop != NULL) {
        if (nn->train->rms_prop->costWeightDerivativeSquaredAccumulated == NULL)
            nn->train->rms_prop->costWeightDerivativeSquaredAccumulated = tensor((void *)self, (tensor_dict){.rank=2, .init=false});
        if (nn->train->rms_prop->costBiasDerivativeSquaredAccumulated == NULL)
            nn->train->rms_prop->costBiasDerivativeSquaredAccumulated = tensor((void *)self, (tensor_dict){.rank=1, .init=false});
    }
    if (nn->train->adam != NULL) {
        if (nn->train->adam->weightsBiasedFirstMomentEstimate == NULL)
            nn->train->adam->weightsBiasedFirstMomentEstimate = nn->tensor((void *)self, (tensor_dict){.rank=2, .init=false});
        if (nn->train->adam->weightsBiasedSecondMomentEstimate == NULL)
            nn->train->adam->weightsBiasedSecondMomentEstimate = nn->tensor((void *)self, (tensor_dict){.rank=2, .init=false});
        if (nn->train->adam->biasesBiasedFirstMomentEstimate == NULL)
            nn->train->adam->biasesBiasedFirstMomentEstimate = nn->tensor((void *)self, (tensor_dict){.rank=1, .init=false});
        if (nn->train->adam->biasesBiasedSecondMomentEstimate == NULL)
            nn->train->adam->biasesBiasedSecondMomentEstimate = nn->tensor((void *)self, (tensor_dict){.rank=1, .init=false});
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
    
    free(nn->weights);
    free(nn->biases);
    
    costWeightDerivativeNode *dcdwTail = nn->networkCostWeightDerivatives;
    while (dcdwTail != NULL && dcdwTail->next ) {
        dcdwTail = dcdwTail->next;
    }
    costWeightDerivativeNode *dcdwNodePt = NULL;
    while (dcdwTail != NULL) {
        dcdwNodePt = dcdwTail->previous;
        free_fmatrix(dcdwTail->dcdw, 0, dcdwTail->m-1, 0, dcdwTail->n-1);
        dcdwTail->dcdw = NULL;
        dcdwTail->next = NULL;
        dcdwTail->previous = NULL;
        free(dcdwTail);
        dcdwTail = dcdwNodePt;
    }
    
    costWeightDerivativeNode *delta_dcdwTail = nn->deltaNetworkCostWeightDerivatives;
    while (delta_dcdwTail != NULL && delta_dcdwTail->next ) {
        delta_dcdwTail = delta_dcdwTail->next;
    }
    costWeightDerivativeNode *delta_dcdwNodePt = NULL;
    while (delta_dcdwTail != NULL) {
        delta_dcdwNodePt = delta_dcdwTail->previous;
        free_fmatrix(delta_dcdwTail->dcdw, 0, delta_dcdwTail->m-1, 0, delta_dcdwTail->n-1);
        delta_dcdwTail->dcdw = NULL;
        delta_dcdwTail->next = NULL;
        delta_dcdwTail->previous = NULL;
        free(delta_dcdwTail);
        delta_dcdwTail = delta_dcdwNodePt;
    }
    
    costBiaseDerivativeNode *dcdbTail = nn->networkCostBiaseDerivatives;
    while (dcdbTail != NULL && dcdbTail->next != NULL) {
        dcdbTail = dcdbTail->next;
    }
    costBiaseDerivativeNode *dcdbNodePt = NULL;
    while (dcdbTail != NULL) {
        dcdbNodePt = dcdbTail->previous;
        free_fvector(dcdbTail->dcdb, 0, dcdbTail->n);
        dcdbTail->dcdb = NULL;
        dcdbTail->next = NULL;
        dcdbTail->previous = NULL;
        free(dcdbTail);
        dcdbTail = dcdbNodePt;
    }
    
    costBiaseDerivativeNode *delta_dcdbTail = nn->deltaNetworkCostBiaseDerivatives;
    while (delta_dcdbTail != NULL && delta_dcdbTail->next != NULL) {
        delta_dcdbTail = delta_dcdbTail->next;
    }
    costBiaseDerivativeNode *delta_dcdbNodePt = NULL;
    while (delta_dcdbTail != NULL) {
        delta_dcdbNodePt = delta_dcdbTail->previous;
        free_fvector(delta_dcdbTail->dcdb, 0, delta_dcdbTail->n);
        delta_dcdbTail->dcdb = NULL;
        delta_dcdbTail->next = NULL;
        delta_dcdbTail->previous = NULL;
        free(delta_dcdbTail);
        delta_dcdbTail = delta_dcdbNodePt;
    }
    
    activationNode *aTail = nn->networkActivations;
    while (aTail != NULL && aTail->next != NULL) {
        aTail = aTail->next;
    }
    activationNode *aNodePt = NULL;
    while (aTail != NULL) {
        aNodePt = aTail->previous;
        free_fvector(aTail->a, 0, aTail->n);
        aTail->a = NULL;
        aTail->next = NULL;
        aTail->previous = NULL;
        free(aTail);
        aTail = aNodePt;
    }
    
    affineTransformationNode *zTail = nn->networkAffineTransformations;
    while (zTail != NULL && zTail->next != NULL) {
        zTail = zTail->next;
    }
    affineTransformationNode *zNodePt = NULL;
    while (zTail != NULL) {
        zNodePt = zTail->previous;
        free_fvector(zTail->z, 0, zTail->n);
        zTail->z = NULL;
        zTail->next = NULL;
        zTail->previous = NULL;
        free(zTail);
        zTail = zNodePt;
    }
    
    if (nn->gpu != NULL) {
        nn->gpu->nullify();
        free(nn->gpu);
    }
    
    if (nn->train->gradient_descent != NULL) {
        free(nn->train->gradient_descent);
    }
    if (nn->train->momentum != NULL) {
        if (nn->weightsVelocity != NULL) free(nn->weightsVelocity);
        if (nn->biasesVelocity != NULL) free(nn->biasesVelocity);
        free(nn->train->momentum);
    }
    if (nn->train->ada_grad != NULL) {
        if (nn->train->ada_grad->costWeightDerivativeSquaredAccumulated != NULL) free(nn->train->ada_grad->costWeightDerivativeSquaredAccumulated);
        if (nn->train->ada_grad->costBiasDerivativeSquaredAccumulated != NULL) free(nn->train->ada_grad->costBiasDerivativeSquaredAccumulated);
        free(nn->train->ada_grad);
    }
    if (nn->train->rms_prop != NULL) {
        if (nn->train->rms_prop->costWeightDerivativeSquaredAccumulated != NULL) free(nn->train->rms_prop->costWeightDerivativeSquaredAccumulated);
        if (nn->train->rms_prop->costBiasDerivativeSquaredAccumulated != NULL) free(nn->train->rms_prop->costBiasDerivativeSquaredAccumulated);
        free(nn->train->rms_prop);
    }
    if (nn->train->adam != NULL) {
        if (nn->train->adam->weightsBiasedFirstMomentEstimate != NULL) free(nn->train->adam->weightsBiasedFirstMomentEstimate);
        if (nn->train->adam->weightsBiasedSecondMomentEstimate != NULL) free(nn->train->adam->weightsBiasedSecondMomentEstimate);
        
        if (nn->train->adam->biasesBiasedFirstMomentEstimate != NULL) free(nn->train->adam->biasesBiasedFirstMomentEstimate);
        if (nn->train->adam->biasesBiasedSecondMomentEstimate != NULL) free(nn->train->adam->biasesBiasedSecondMomentEstimate);
        free(nn->train->adam);
    }
    free(nn->train);
}

static void gpu_alloc(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->gpu = metalCompute();
}

static int eval(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    float result = 0.0f;
    activationNode *aNodePt = NULL;
    
    int sum = 0;
    for (int k=0; k<nn->data->test->m; k++) {
        
        aNodePt = nn->networkActivations;
        for (int i=0; i<nn->number_of_features; i++) {
            aNodePt->a[i] = nn->data->test->set[k][i];
        }

        feedforward(self);
        
        aNodePt = nn->networkActivations;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
        
        result = (float)argmax(aNodePt->a, aNodePt->n);
        sum = sum + (result == nn->data->test->set[k][nn->number_of_features]);
    }
    
    return sum;
}

static int evaluate(void * _Nonnull self, bool metal) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    int sum = 0;
    double rt = 0.0;
    
#ifdef __APPLE__
    if (metal) {
        unsigned int weightsTableSize = 0;
        unsigned int biasesTableSize = 0;
        for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
            weightsTableSize = weightsTableSize + (nn->weightsDimensions[l].m * nn->weightsDimensions[l].n);
            biasesTableSize = biasesTableSize + nn->biasesDimensions[l].n;
        }
        
        nn->gpu->allocate_buffers((void *)nn);
        nn->gpu->prepare("feedforward");
        nn->gpu->format_data(nn->data->test->set, nn->data->test->m, nn->number_of_features);
        
        float result[nn->data->test->m];
        rt = realtime();
        
        nn->gpu->feedforward((void *)nn, result);
        float vector_sum = 0.0;
        vDSP_sve(result, 1, &vector_sum, nn->data->test->m);
        sum = (int)vector_sum;
        
        rt = realtime() - rt;
        
    } else {
        rt = realtime();
        sum = eval(self);
        rt = realtime() - rt;
    }
#else
    rt = realtime();
    sum = eval(self);
    rt = realtime() - rt;
#endif
    
    fprintf(stdout, "%s: total infer time in evaluation for %u input test data (s): %f\n", DEFAULT_CONSOLE_WRITER, nn->data->test->m, rt);
    
    return sum;
}

//
//  Compute the total cost function using a cross-entropy formulation
//
static float totalCost(void * _Nonnull self, float * _Nonnull * _Nonnull data, unsigned int m, bool convert) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    float norm, sum;
    activationNode *aNodePt = NULL;
    
    float cost = 0.0f;
    for (int i=0; i<m; i++) {

        aNodePt = nn->networkActivations;
        for (int j=0; j<nn->number_of_features; j++) {
            aNodePt->a[j] = data[i][j];
        }
        
        feedforward(self);
        aNodePt = nn->networkActivations;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
        
        float y[aNodePt->n];
        memset(y, 0.0f, sizeof(y));
        if (convert == true) {
            for (int j=0; j<aNodePt->n; j++) {
                if (data[i][nn->number_of_features] == nn->parameters->classifications[j]) {
                    y[j] = 1.0f;
                }
            }
        } else {
            int idx = (int)nn->number_of_features;
            for (int j=0; j<aNodePt->n; j++) {
                y[j] = data[i][idx];
                idx++;
            }
        }
        cost = cost + crossEntropyCost(aNodePt->a, y, aNodePt->n) / m;
        
        sum = 0.0f;
        unsigned int stride = 0;
        for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
            unsigned int m = nn->weightsDimensions[l].m;
            unsigned int n = nn->weightsDimensions[l].n;
            norm = frobeniusNorm(nn->weights+stride, (m * n));
            sum = sum + (norm*norm);
            stride = stride + (m * n);
        }
        cost = cost + 0.5f*(nn->parameters->lambda/(float)m)*sum;
    }
    
    return cost;
}
