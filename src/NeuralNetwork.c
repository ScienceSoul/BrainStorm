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
#include "NetworkUtils.h"
//#include "Data.h"
#include "TimeProfile.h"

static void initNeuralData(void * _Nonnull self);

static void genesis(void * _Nonnull self);
static void finale(void * _Nonnull self);
static void gpu_alloc(void * _Nonnull self);

static void trainNetwork(void * _Nonnull self, bool metal, bool * _Nullable showTotalCost);

static void miniBatch(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch);

static void updateWeightsBiases(void * _Nonnull self);

static void batchAccumulation(void * _Nonnull self);

static void * _Nullable backpropagation(void * _Nonnull self);

static int evaluate(void * _Nonnull self, bool metal);

static float totalCost(void * _Nonnull self, float * _Nonnull * _Nonnull data, unsigned int m, bool convert);

static void feedforward(void * _Nonnull self);


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
    *nn = (NeuralNetwork){.weights=NULL, .weightsVelocity=NULL, .biases=NULL, .biasesVelocity=NULL, .networkActivations=NULL, .networkAffineTransformations=NULL, .networkCostWeightDerivatives=NULL, .networkCostBiaseDerivatives=NULL, .deltaNetworkCostWeightDerivatives=NULL, .deltaNetworkCostBiaseDerivatives=NULL, .gpu=NULL, .adaGrad=NULL, .rmsProp=NULL, .adam=NULL};
    
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
    strcpy(nn->parameters->supported_parameters[9], "regularization_factor");
    strcpy(nn->parameters->supported_parameters[10], "momentum");
    strcpy(nn->parameters->supported_parameters[11], "adagrad");
    strcpy(nn->parameters->supported_parameters[12], "rmsprop");
    strcpy(nn->parameters->supported_parameters[13], "adam");
    
    bzero(nn->parameters->data, MAX_LONG_STRING_LENGTH);
    strcpy(nn->parameters->dataName, "<empty>");
    nn->parameters->epochs = 30;
    nn->parameters->miniBatchSize = 10;
    nn->parameters->eta = 0.5f;
    nn->parameters->lambda = 0.0f;
    nn->parameters->mu = 0.0f;
    nn->parameters->numberOfLayers = 0;
    nn->parameters->numberOfActivationFunctions = 0;
    nn->parameters->numberOfClassifications = 0;
    nn->adapativeLearningRateMethod = 0;
    memset(nn->parameters->topology, 0, sizeof(nn->parameters->topology));
    memset(nn->parameters->classifications, 0, sizeof(nn->parameters->classifications));
    memset(nn->parameters->split, 0, sizeof(nn->parameters->split));
    
    memset(*nn->parameters->activationFunctions, 0, (MAX_NUMBER_NETWORK_LAYERS*128)*sizeof(char));
    
    for (int i=0; i<MAX_NUMBER_NETWORK_LAYERS; i++) {
        nn->activationFunctions[i] = NULL;
        nn->activationDerivatives[i] = NULL;
    }
    
    nn->constructor = allocateConstructor();
    
    nn->load_params_from_input_file = loadParametersFromImputFile;
    
    nn->genesis = genesis;
    nn->finale = finale;
    nn->gpu_alloc = gpu_alloc;
    nn->train = trainNetwork;
    nn->miniBatch = miniBatch;
    nn->updateWeightsBiases = updateWeightsBiases;
    nn->batchAccumulation = batchAccumulation;
    nn->backpropagation = backpropagation;
    nn->evaluate = evaluate;
    nn->totalCost = totalCost;
    nn->feedforward = feedforward;
    
    return nn;
}

//
//  Create the network layers, i.e. allocates memory for the weight, bias, activation, z, dC/dx and dC/db data structures
//
static void genesis(void * _Nonnull self) {
    
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
    nn->number_of_features = 0;
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
    
    nn->weights = initMatrices(nn->parameters->topology, nn->parameters->numberOfLayers, true);
    nn->weightsVelocity = initMatrices(nn->parameters->topology, nn->parameters->numberOfLayers, true);
    nn->biases = initVectors(nn->parameters->topology, nn->parameters->numberOfLayers, true);
    nn->biasesVelocity = initVectors(nn->parameters->topology, nn->parameters->numberOfLayers, true);
    nn->networkActivations = (activationNode *)initNetworkActivations(nn->parameters->topology, nn->parameters->numberOfLayers);
    nn->networkAffineTransformations = (affineTransformationNode *)initNetworkAffineTransformations(nn->parameters->topology, nn->parameters->numberOfLayers);
    nn->networkCostWeightDerivatives = (costWeightDerivativeNode *)initNetworkCostWeightDerivatives(nn->parameters->topology, nn->parameters->numberOfLayers);
    nn->networkCostBiaseDerivatives = (costBiaseDerivativeNode *)initNetworkCostBiaseDerivatives(nn->parameters->topology, nn->parameters->numberOfLayers);
    nn->deltaNetworkCostWeightDerivatives = (costWeightDerivativeNode *)initNetworkCostWeightDerivatives(nn->parameters->topology, nn->parameters->numberOfLayers);
    nn->deltaNetworkCostBiaseDerivatives = (costBiaseDerivativeNode *)initNetworkCostBiaseDerivatives(nn->parameters->topology, nn->parameters->numberOfLayers);
    
    if (nn->adaGrad != NULL) {
        nn->adaGrad->costWeightDerivativeSquaredAccumulated = initMatrices(nn->parameters->topology, nn->parameters->numberOfLayers, false);
        nn->adaGrad->costBiasDerivativeSquaredAccumulated = initVectors(nn->parameters->topology, nn->parameters->numberOfLayers, false);
    }
    if (nn->rmsProp != NULL) {
        nn->rmsProp->costWeightDerivativeSquaredAccumulated = initMatrices(nn->parameters->topology, nn->parameters->numberOfLayers, false);
        nn->rmsProp->costBiasDerivativeSquaredAccumulated = initVectors(nn->parameters->topology, nn->parameters->numberOfLayers, false);
    }
    if (nn->adam != NULL) {
        nn->adam->weightsBiasedFirstMomentEstimate = initMatrices(nn->parameters->topology, nn->parameters->numberOfLayers, false);
        nn->adam->weightsBiasedSecondMomentEstimate = initMatrices(nn->parameters->topology, nn->parameters->numberOfLayers, false);
        nn->adam->biasesBiasedFirstMomentEstimate = initVectors(nn->parameters->topology, nn->parameters->numberOfLayers, false);
        nn->adam->biasesBiasedSecondMomentEstimate = initVectors(nn->parameters->topology, nn->parameters->numberOfLayers, false);
    }
}

//
// Free-up all the memory used by a network
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
    free(nn->weightsVelocity);
    free(nn->biases);
    free(nn->biasesVelocity);
    
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
    
    if (nn->adaGrad != NULL) {
        if (nn->adaGrad->costWeightDerivativeSquaredAccumulated != NULL) free(nn->adaGrad->costWeightDerivativeSquaredAccumulated);
        if (nn->adaGrad->costBiasDerivativeSquaredAccumulated != NULL) free(nn->adaGrad->costBiasDerivativeSquaredAccumulated);
        free(nn->adaGrad);
    }
    
    if (nn->rmsProp != NULL) {
        if (nn->rmsProp->costWeightDerivativeSquaredAccumulated != NULL) free(nn->rmsProp->costWeightDerivativeSquaredAccumulated);
        if (nn->rmsProp->costBiasDerivativeSquaredAccumulated != NULL) free(nn->rmsProp->costBiasDerivativeSquaredAccumulated);
        free(nn->rmsProp);
    }
    if (nn->adam != NULL) {
        if (nn->adam->weightsBiasedFirstMomentEstimate != NULL) free(nn->adam->weightsBiasedFirstMomentEstimate);
        if (nn->adam->weightsBiasedSecondMomentEstimate != NULL) free(nn->adam->weightsBiasedSecondMomentEstimate);
        
        if (nn->adam->biasesBiasedFirstMomentEstimate != NULL) free(nn->adam->biasesBiasedFirstMomentEstimate);
        if (nn->adam->biasesBiasedSecondMomentEstimate != NULL) free(nn->adam->biasesBiasedSecondMomentEstimate);
    }
}

static void gpu_alloc(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->gpu = metalCompute();
}

static void trainNetwork(void * _Nonnull self, bool metal, bool * _Nullable showTotalCost) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    nn->number_of_features = nn->parameters->topology[0];
    
    fprintf(stdout, "%s: train neural network with the %s data set.\n", DEFAULT_CONSOLE_WRITER, nn->parameters->dataName);
    
    // Stochastic gradient descent
    float **miniBatch = floatmatrix(0, nn->parameters->miniBatchSize-1, 0, nn->data->training->n-1);
    nn->batch = miniBatch;
    int delta;
    for (int k=1; k<=nn->parameters->epochs; k++) {
        delta = 0;
        shuffle(nn->data->training->set, nn->data->training->m, nn->data->training->n);
        
        fprintf(stdout, "%s: Epoch {%d/%d}:\n", DEFAULT_CONSOLE_WRITER, k, nn->parameters->epochs);
        double train_time = 0.0;
        const int percentPrint = 5;
        int train_size = (int)nn->data->training->m/nn->parameters->miniBatchSize;
        int step = train_size / (100/percentPrint);
        int nextPrint = step;
        int i = 0;
        for (int l=1; l<=(int)ceil((int)nn->data->training->m/nn->parameters->miniBatchSize); l++) {
            memcpy(*miniBatch, *nn->data->training->set+delta, (nn->parameters->miniBatchSize*(int)nn->data->training->n)*sizeof(float));
            double rt = realtime();
            nn->miniBatch((void *)nn, miniBatch);
            rt = realtime() - rt;
            train_time += rt;
            delta = delta + (nn->parameters->miniBatchSize*(int)nn->data->training->n);
            
            i++;
            if (i >= nextPrint) {
                int percent = (100 * i) / train_size;
                fprintf(stdout, "...%d%%\n", percent);
                fflush(stdout);
                nextPrint += step;
            }
        }
        fprintf(stdout, "%s: time to complete all training data set (s): %f\n", DEFAULT_CONSOLE_WRITER, train_time);
        
        if (nn->data->test->set != NULL) {
            fprintf(stdout, "%s: Epoch {%d/%d}: testing network with {%u} inputs:\n", DEFAULT_CONSOLE_WRITER, k, nn->parameters->epochs, nn->data->test->m);
            int result = nn->evaluate(self, metal);
            fprintf(stdout, "%s: Epoch {%d/%d}: {%d} / {%u}.\n", DEFAULT_CONSOLE_WRITER, k, nn->parameters->epochs, result, nn->data->test->m);
        }
        
        if (showTotalCost != NULL) {
            if (*showTotalCost == true) {
                double rt = realtime();
                float cost = nn->totalCost(self, nn->data->training->set, nn->data->training->m, false);
                rt = realtime() -  rt;
                fprintf(stdout, "%s: cost on training data: {%f} / Time (s): %f\n", DEFAULT_CONSOLE_WRITER, cost, rt);
                
                if (nn->data->test->set != NULL) {
                    double rt = realtime();
                    cost = nn->totalCost(self, nn->data->test->set, nn->data->test->m, true);
                    rt = realtime() -  rt;
                    fprintf(stdout, "%s: cost on test data: {%f} / Time (s): %f\n", DEFAULT_CONSOLE_WRITER, cost, rt);
                }
            }
        }
        fprintf(stdout, "\n");
    }

    free_fmatrix(miniBatch, 0, nn->parameters->miniBatchSize-1, 0, nn->data->training->n-1);
    fprintf(stdout, "%s: all done.\n", DEFAULT_CONSOLE_WRITER);
}

static void miniBatch(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    costWeightDerivativeNode *dcdwNodePt = nn->networkCostWeightDerivatives;
    while (dcdwNodePt != NULL) {
        memset(*dcdwNodePt->dcdw, 0.0f, (dcdwNodePt->m*dcdwNodePt->n)*sizeof(float));
        dcdwNodePt = dcdwNodePt->next;
    }
    
    costBiaseDerivativeNode *dcdbNodePt = nn->networkCostBiaseDerivatives;
    while (dcdbNodePt != NULL) {
        memset(dcdbNodePt->dcdb, 0.0f, dcdbNodePt->n*sizeof(float));
        dcdbNodePt = dcdbNodePt->next;
    }
    
    costWeightDerivativeNode *delta_dcdwNodePt = nn->deltaNetworkCostWeightDerivatives;
    while (delta_dcdwNodePt != NULL) {
        memset(*delta_dcdwNodePt->dcdw, 0.0f, (delta_dcdwNodePt->m*delta_dcdwNodePt->n)*sizeof(float));
        delta_dcdwNodePt = delta_dcdwNodePt->next;
    }
    
    costBiaseDerivativeNode *delta_dcdbNodePt = nn->deltaNetworkCostBiaseDerivatives;
    while (delta_dcdbNodePt != NULL) {
        memset(delta_dcdbNodePt->dcdb, 0.0f, delta_dcdbNodePt->n*sizeof(float));
        delta_dcdbNodePt = delta_dcdbNodePt->next;
    }
    
    double rt = realtime();
    for (int i=0; i<nn->parameters->miniBatchSize; i++) {
        nn->example_idx = i;
        nn->backpropagation((void *)nn);
        nn->batchAccumulation((void *)nn);
    }
    rt = realtime() - rt;
#ifdef VERBOSE
    fprintf(stdout, "%s: time to complete a single mini-batch (s): %f\n", DEFAULT_CONSOLE_WRITER, rt);
#endif
    
    nn->updateWeightsBiases((void *)nn);
}

static void batchAccumulation(void * _Nonnull self) {
    
    // Accumulate dcdw and dc/db
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    costWeightDerivativeNode *dcdwNodePt = nn->networkCostWeightDerivatives;
    costBiaseDerivativeNode *dcdbNodePt = nn->networkCostBiaseDerivatives;
    costWeightDerivativeNode *delta_dcdwNodePt = nn->deltaNetworkCostWeightDerivatives;
    costBiaseDerivativeNode *delta_dcdbNodePt = nn->deltaNetworkCostBiaseDerivatives;
    while (dcdwNodePt != NULL && delta_dcdwNodePt != NULL) {
        for (int i=0; i<dcdwNodePt->m; i++) {
            for (int j=0; j<dcdwNodePt->n; j++) {
                dcdwNodePt->dcdw[i][j] = dcdwNodePt->dcdw[i][j] + delta_dcdwNodePt->dcdw[i][j];
            }
        }
        for (int i=0; i<dcdbNodePt->n; i++) {
            dcdbNodePt->dcdb[i] = dcdbNodePt->dcdb[i] + delta_dcdbNodePt->dcdb[i];
        }
        dcdwNodePt = dcdwNodePt->next;
        dcdbNodePt = dcdbNodePt->next;
        delta_dcdwNodePt = delta_dcdwNodePt->next;
        delta_dcdbNodePt = delta_dcdbNodePt->next;
    }
}

static void updateWeightsBiases(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // Update weights
    unsigned int stride = 0;
    unsigned int l = 0;
    costWeightDerivativeNode *dcdwNodePt = nn->networkCostWeightDerivatives;
    while (dcdwNodePt != NULL) {
        unsigned int m = nn->weightsDimensions[l].m;
        unsigned int n = nn->weightsDimensions[l].n;
        
        // Adpative learning rate
        float coeff[m][n];
        
/////////////////////////////////////////// AdaGrad learning rate ////////////////////////////////////////////////////////
        if (nn->adapativeLearningRateMethod == ADAGRAD) {
            for (int i=0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    nn->adaGrad->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)] = nn->adaGrad->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)] + ( ((1.0f/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j]) * ((1.0f/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j]) );
                }
            }
            for (int i=0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    coeff[i][j] = ( nn->parameters->eta/(nn->adaGrad->delta+sqrtf(nn->adaGrad->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)])) ) * ( (1.0f/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j] );
                }
            }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            
/////////////////////////////////////////// RMSProp learning rate ////////////////////////////////////////////////////////
        } else if (nn->adapativeLearningRateMethod == RMSPROP) {
            for (int i=0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    nn->rmsProp->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)] = (nn->rmsProp->decayRate*nn->rmsProp->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)]) + (1.0f-nn->rmsProp->decayRate)*( ((1.0f/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j]) * ((1.0f/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j]) );
                }
            }
            for (int i=0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    coeff[i][j] = ( nn->parameters->eta/(sqrtf(nn->rmsProp->delta+nn->rmsProp->costWeightDerivativeSquaredAccumulated[stride+((i*n)+j)])) ) * ( (1.0f/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j] );
                }
            }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            
/////////////////////////////////////////// Adam learning rate ///////////////////////////////////////////////////////////
        } else if (nn->adapativeLearningRateMethod == ADAM) {
            nn->adam->time++;
            float s_hat[m][n];
            float r_hat[m][n];
            for (int i=0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    // Update biased first moment estimate
                    nn->adam->weightsBiasedFirstMomentEstimate[stride+((i*n)+j)] = (nn->adam->decayRate1*nn->adam->weightsBiasedFirstMomentEstimate[stride+((i*n)+j)]) + (1.0f-nn->adam->decayRate1)*( ((1.0f/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j]) );
                    // Update biased second moment estimate
                    nn->adam->weightsBiasedSecondMomentEstimate[stride+((i*n)+j)] = (nn->adam->decayRate2*nn->adam->weightsBiasedSecondMomentEstimate[stride+((i*n)+j)]) + (1.0f-nn->adam->decayRate2)*( ((1.0f/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j]) * ((1.0f/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j]) );
                    
                    // Correct bias in first moment
                    s_hat[i][j] = nn->adam->weightsBiasedFirstMomentEstimate[stride+((i*n)+j)] / (1.0f - powf(nn->adam->decayRate1, (float)nn->adam->time));
                    // Correct bias in second moment
                    r_hat[i][j] = nn->adam->weightsBiasedSecondMomentEstimate[stride+((i*n)+j)] / (1.0f - powf(nn->adam->decayRate2, (float)nn->adam->time));
                }
            }
            for (int i=0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    coeff[i][j] = nn->adam->stepSize*( s_hat[i][j] / (sqrtf(r_hat[i][j])+nn->adam->delta) );
                }
            }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            
        } else {
            for (int i=0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    coeff[i][j] = (nn->parameters->eta/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j];
                }
            }
        }
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->weightsVelocity[stride+((i*n)+j)] = nn->parameters->mu*nn->weightsVelocity[stride+((i*n)+j)] - coeff[i][j];
                nn->weights[stride+((i*n)+j)] = (1.0f-((nn->parameters->eta*nn->parameters->lambda)/(float)nn->data->training->m))*nn->weights[stride+((i*n)+j)] + nn->weightsVelocity[stride+((i*n)+j)];
            }
        }
        dcdwNodePt = dcdwNodePt->next;
        l++;
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    l = 0;
    costBiaseDerivativeNode *dcdbNodePt = nn->networkCostBiaseDerivatives;
    while (dcdbNodePt != NULL) {
        unsigned int n = nn->biasesDimensions[l].n;
        
        // Adpative learning rate
        float coeff[n];
        
/////////////////////////////////////////// AdaGrad learning rate ////////////////////////////////////////////////////////
        if (nn->adapativeLearningRateMethod == ADAGRAD) {
            for (int i=0; i<n; i++) {
                nn->adaGrad->costBiasDerivativeSquaredAccumulated[stride+i] = nn->adaGrad->costBiasDerivativeSquaredAccumulated[stride+i] + ( ((1.0f/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i]) * ((1.0f/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i]) );
            }
            for (int i=0; i<n; i++) {
                coeff[i] = ( nn->parameters->eta/(nn->adaGrad->delta+sqrtf(nn->adaGrad->costBiasDerivativeSquaredAccumulated[stride+i])) ) * ( ((1.0f/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i]) );
            }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            
/////////////////////////////////////////// RMSProp learning rate ////////////////////////////////////////////////////////
        } else if (nn->adapativeLearningRateMethod == RMSPROP) {
            for (int i=0; i<n; i++) {
                nn->rmsProp->costBiasDerivativeSquaredAccumulated[stride+i] = (nn->rmsProp->decayRate*nn->rmsProp->costBiasDerivativeSquaredAccumulated[stride+i]) + (1.0-nn->rmsProp->decayRate)*( ((1.0f/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i]) * ((1.0f/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i]) );
            }
            for (int i=0; i<n; i++) {
                coeff[i] = ( nn->parameters->eta/(sqrtf(nn->rmsProp->delta+nn->rmsProp->costBiasDerivativeSquaredAccumulated[stride+i])) ) * ( ((1.0f/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i]) );
            }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            
/////////////////////////////////////////// Adam learning rate ///////////////////////////////////////////////////////////
        } else if (nn->adapativeLearningRateMethod == ADAM) {
            float s_hat[n];
            float r_hat[n];
            for (int i=0; i<n; i++) {
                // Update biased first moment estimate
                nn->adam->biasesBiasedFirstMomentEstimate[stride+i] = (nn->adam->decayRate1*nn->adam->biasesBiasedFirstMomentEstimate[stride+i]) + (1.0-nn->adam->decayRate1)*( ((1.0f/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i]) );
                // Update biased second moment estimate
                nn->adam->biasesBiasedSecondMomentEstimate[stride+i] = (nn->adam->decayRate2*nn->adam->biasesBiasedSecondMomentEstimate[stride+i]) + (1.0-nn->adam->decayRate2)*( ((1.0f/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i]) * ((1.0f/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i]) );
                
                // Correct bias in first moment
                s_hat[i] = nn->adam->biasesBiasedFirstMomentEstimate[stride+i] / (1.0f - powf(nn->adam->decayRate1, (float)nn->adam->time));
                // Correct bias in second moment
                r_hat[i] = nn->adam->biasesBiasedSecondMomentEstimate[stride+i] / (1.0f - powf(nn->adam->decayRate2, (float)nn->adam->time));
            }
            for (int i=0; i<n; i++) {
                coeff[i] = nn->adam->stepSize*( s_hat[i] / (sqrtf(r_hat[i])+nn->adam->delta) );
            }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
        } else {
            for (int i=0; i<n; i++) {
                coeff[i] = (nn->parameters->eta/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i];
            }
        }
        
        for (int i=0; i<n; i++) {
            nn->biasesVelocity[stride+i] = nn->parameters->mu*nn->biasesVelocity[stride+i] - coeff[i];
            nn->biases[stride+i] = nn->biases[stride+i] + nn->biasesVelocity[stride+i];
        }
        dcdbNodePt = dcdbNodePt->next;
        l++;
        stride = stride + n;
    }
}

//
//  Return the gradient of the cross-entropy cost function C_x layers by layers
//
static void * _Nullable backpropagation(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // Activations at the input layer
    activationNode *aNodePt = nn->networkActivations;
    for (int i=0; i<nn->number_of_features; i++) {
        aNodePt->a[i] = nn->batch[nn->example_idx][i];
    }
    
    // Feedforward
    feedforward(nn);
    
    // ------------- Backward pass
    // At last layer
    
    activationNode *aTail = nn->networkActivations;
    while (aTail != NULL && aTail->next != NULL) {
        aTail = aTail->next;
    }
    affineTransformationNode *zTail = nn->networkAffineTransformations;
    while (zTail != NULL && zTail->next != NULL) {
        zTail = zTail->next;
    }
    
    float delta[nn->max_number_of_nodes_in_layer];
    float buffer[nn->max_number_of_nodes_in_layer];
    memset(delta, 0.0f, sizeof(delta));
    memset(buffer, 0.0f, sizeof(buffer));
    
    // Compute delta
    int k = (int)nn->number_of_features;
    for (int i=0; i<aTail->n; i++) {
        delta[i] = aTail->a[i] - nn->batch[nn->example_idx][k];
        k++;
    }
    
    //dc/dw and dc/db at last layer
    costWeightDerivativeNode *dcdwTail = nn->deltaNetworkCostWeightDerivatives;
    while (dcdwTail != NULL && dcdwTail->next != NULL) {
        dcdwTail = dcdwTail->next;
    }
    costBiaseDerivativeNode *dcdbTail = nn->deltaNetworkCostBiaseDerivatives;
    while (dcdbTail != NULL && dcdbTail->next != NULL) {
        dcdbTail = dcdbTail->next;
    }
    aNodePt = aTail->previous;
    for (int i=0; i<dcdwTail->m; i++) {
        for (int j=0; j<dcdwTail->n; j++) {
            dcdwTail->dcdw[i][j] = aNodePt->a[j]*delta[i];
        }
    }
    for (int i=0; i<dcdbTail->n; i++) {
        dcdbTail->dcdb[i] = delta[i];
    }
    
    // The backward pass loop
    
    // Stride to weithts at last layer
    unsigned int stride = 0;
    unsigned int m, n;
    for (int l=0; l<nn->parameters->numberOfLayers-2; l++) {
        m = nn->weightsDimensions[l].m;
        n = nn->weightsDimensions[l].n;
        stride = stride + (m * n);
    }
    
    affineTransformationNode *zNodePt = zTail->previous;
    costWeightDerivativeNode *dcdwNodePt = dcdwTail->previous;
    costBiaseDerivativeNode *dcdbNodePt = dcdbTail->previous;
    
    unsigned int l = nn->parameters->numberOfLayers - 2;
    while (dcdwNodePt != NULL && dcdbNodePt != NULL) {
        aNodePt = aNodePt->previous;
        
        float sp[zNodePt->n];
        for (int i=0; i<zNodePt->n; i++) {
            sp[i] = nn->activationDerivatives[l-1](zNodePt->z[i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->weightsDimensions[l].m, (int)nn->weightsDimensions[l].n, 1.0, nn->weights+stride, (int)nn->weightsDimensions[l].n, delta, 1, 0.0, buffer, 1);
        for (int i=0; i<zNodePt->n; i++) {
            delta[i] = buffer[i] * sp[i];
        }
        // dc/dw at layer l
        for (int i=0; i<dcdwNodePt->m; i++) {
            for (int j=0; j<dcdwNodePt->n; j++) {
                dcdwNodePt->dcdw[i][j] = aNodePt->a[j]*delta[i];
            }
        }
        // dc/db at layer l
        for (int i=0; i<dcdbNodePt->n; i++) {
            dcdbNodePt->dcdb[i] = delta[i];
        }
        
        zNodePt = zNodePt->previous;
        dcdwNodePt = dcdwNodePt->previous;
        dcdbNodePt = dcdbNodePt->previous;
        stride = stride - (nn->weightsDimensions[l-1].m * nn->weightsDimensions[l-1].n);
        l--;
    }
    
    return NULL;
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

        nn->feedforward(self);
        
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
        
        nn->feedforward(self);
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

//
//  Return the output of the network for a given activation input
//
static void feedforward(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;

    activationNode *aNodePt = nn->networkActivations;
    affineTransformationNode *zNodePt = nn->networkAffineTransformations;
    
    unsigned int stride1 = 0;
    unsigned int stride2 = 0;
    for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
        unsigned int m = nn->weightsDimensions[l].m;
        unsigned int n = nn->weightsDimensions[l].n;
        
        aNodePt = aNodePt->next;
        zNodePt = zNodePt->next;
        float buffer[aNodePt->n];
        memset(buffer, 0.0f, sizeof(buffer));
        
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n, 1.0, nn->weights+stride1, (int)n, aNodePt->previous->a, 1, 0.0, buffer, 1);
#ifdef __APPLE__
        vDSP_vadd(buffer, 1, nn->biases+stride2, 1, zNodePt->z, 1,nn->biasesDimensions[l].n);
#else
        for (int i=0; i<nn->biasesDimensions[l].n; i++) {
            zNodePt->z[i] = buffer[i] + nn->biases[stride2+i];
        }
#endif
        float *vec = NULL;
        unsigned int *vec_length = NULL;
        if (strcmp(nn->parameters->activationFunctions[l], "softmax") == 0) {
            vec = zNodePt->z;
            vec_length = &zNodePt->n;
        }
        for (int i=0; i<aNodePt->n; i++) {
            aNodePt->a[i] = nn->activationFunctions[l](zNodePt->z[i],vec, vec_length);
        }
        nanToNum(aNodePt->a, aNodePt->n);
        
        stride1 = stride1 + (m * n);
        stride2 = stride2 + nn->biasesDimensions[l].n;
    }
}
