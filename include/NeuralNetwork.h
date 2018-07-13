//
//  NeuralNetwork.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 31/05/2017.
//

#ifndef NeuralNetwork_h
#define NeuralNetwork_h

#include "Data.h"
#include "NetworkParams.h"
#include "NetworkConstructor.h"
#include "NetworkUtils.h"
#include "MetalCompute.h"
#include "Optimizers.h"
#include "NetworkPrimitiveFunctions.h"

typedef struct weightMatrixDimension {
    unsigned int m, n;
} weightMatrixDimension;

typedef struct biasVectorDimension {
    unsigned int n;
} biasVectorDimension;

typedef struct activationNode {
    unsigned int n;
    float * _Nullable a;
    struct activationNode * _Nullable next;
    struct activationNode * _Nullable previous;
} activationNode;

typedef struct affineTransformationNode {
    unsigned int n;
    float * _Nullable z;
    struct affineTransformationNode * _Nullable next;
    struct affineTransformationNode * _Nullable previous;
} affineTransformationNode;

typedef struct costWeightDerivativeNode {
    unsigned int m, n;
    float * _Nullable * _Nullable dcdw;
    struct costWeightDerivativeNode * _Nullable next;
    struct costWeightDerivativeNode * _Nullable previous;
} costWeightDerivativeNode;

typedef struct costBiaseDerivativeNode {
    unsigned int n;
    float * _Nullable dcdb;
    struct costBiaseDerivativeNode * _Nullable next;
    struct costBiaseDerivativeNode * _Nullable previous;
} costBiaseDerivativeNode;

typedef struct Train {
    GradientDescentOptimizer * _Nullable gradient_descent;
    MomentumOptimizer * _Nullable momentum;
    AdaGradOptimizer * _Nullable ada_grad;
    RMSPropOptimizer * _Nullable rms_prop;
    AdamOptimizer * _Nullable adam;
    void (* _Nullable next_batch)(void * _Nonnull neural, float * _Nonnull * _Nonnull placeholder, unsigned int batchSize);
    int (* _Nullable batch_range)(void * _Nonnull neural, unsigned int batchSize);
    void (* _Nullable progression)(void * _Nonnull neural, progress_dict progress_dict);
} Train;

typedef struct NeuralNetwork {
    
    data * _Nullable data;
    float * _Nullable * _Nullable batch;
    networkConstructor * _Nullable constructor;
    networkParameters * _Nullable parameters;
    int (* _Nullable load_params_from_input_file)(void * _Nonnull self, const char * _Nonnull paraFile);
    
    int example_idx;
    unsigned int number_of_parameters;
    unsigned int number_of_features;
    unsigned int max_number_of_nodes_in_layer;
    
    float * _Nullable weights;
    float * _Nullable weightsVelocity;
    float * _Nullable biases;
    float * _Nullable biasesVelocity;
    weightMatrixDimension weightsDimensions[MAX_NUMBER_NETWORK_LAYERS];
    biasVectorDimension biasesDimensions[MAX_NUMBER_NETWORK_LAYERS];
    
    activationNode * _Nullable networkActivations;
    affineTransformationNode * _Nullable networkAffineTransformations;
    costWeightDerivativeNode * _Nullable networkCostWeightDerivatives;
    costBiaseDerivativeNode * _Nullable networkCostBiaseDerivatives;
    costWeightDerivativeNode * _Nullable deltaNetworkCostWeightDerivatives;
    costBiaseDerivativeNode * _Nullable deltaNetworkCostBiaseDerivatives;
    
    Train * _Nullable train;
    MetalCompute * _Nullable gpu;
    
    void (* _Nullable genesis)(void * _Nonnull self, char * _Nonnull init_stategy);
    void (* _Nullable finale)(void * _Nonnull self);
    float * _Nonnull (* _Nullable tensor)(void * _Nonnull self, tensor_dict tensor_dict);
    void (* _Nullable gpu_alloc)(void * _Nonnull self);
    
    float (* _Nonnull activationFunctions[MAX_NUMBER_NETWORK_LAYERS])(float z, float * _Nullable vec, unsigned int * _Nullable n);
    float (* _Nonnull activationDerivatives[MAX_NUMBER_NETWORK_LAYERS])(float z);
    int (* _Nullable evaluate)(void * _Nonnull self, bool metal);
    float (* _Nullable totalCost)(void * _Nonnull self, float * _Nonnull * _Nonnull data, unsigned int m, bool convert);
    
    float (* _Nullable l0_regularizer)(void * _Nonnull neural, int i, int j, int n, int stride);
    float (* _Nullable l1_regularizer)(void * _Nonnull neural, int i, int j, int n, int stride);
    float (* _Nullable l2_regularizer)(void * _Nonnull neural, int i, int j, int n, int stride);
    float (* _Nullable regularizer[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, int i, int j, int n, int stride);
    
} NeuralNetwork;

NeuralNetwork * _Nonnull newNeuralNetwork(void);

#endif /* NeuralNetwork_h */

