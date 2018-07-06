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
#include "MetalCompute.h"
#include "Utils.h"
#include "Optimization.h"

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
    unsigned int adapativeLearningRateMethod;
    
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
    
    MetalCompute * _Nullable gpu;
    
    AdaGrad * _Nullable adaGrad;
    RMSProp * _Nullable rmsProp;
    Adam    * _Nullable adam;
    
    void (* _Nullable genesis)(void * _Nonnull self);
    void (* _Nullable finale)(void * _Nonnull self);
    void (* _Nullable gpu_alloc)(void * _Nonnull self);
    
    void (* _Nullable train)(void * _Nonnull self, bool metal, bool * _Nullable showTotalCost);
    void (* _Nullable miniBatch)(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch);
    void (* _Nullable updateWeightsBiases)(void * _Nonnull self);
    void (* _Nullable batchAccumulation)(void * _Nonnull self);
    void * _Nullable (* _Nullable backpropagation)(void * _Nonnull self);
    void (* _Nonnull feedforward)(void * _Nonnull self);
    float (* _Nonnull activationFunctions[MAX_NUMBER_NETWORK_LAYERS])(float z, float * _Nullable vec, unsigned int * _Nullable n);
    float (* _Nonnull activationDerivatives[MAX_NUMBER_NETWORK_LAYERS])(float z);
    int (* _Nullable evaluate)(void * _Nonnull self, bool metal);
    float (* _Nullable totalCost)(void * _Nonnull self, float * _Nonnull * _Nonnull data, unsigned int m, bool convert);
} NeuralNetwork;

NeuralNetwork * _Nonnull newNeuralNetwork(void);

#endif /* NeuralNetwork_h */

