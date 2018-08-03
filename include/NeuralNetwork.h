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
#include "Optimizers.h"
#include "NetworkOps.h"

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

typedef struct dense_network {
    unsigned int num_dense_layers;
    
    tensor * _Nullable weights;
    tensor * _Nullable weightsVelocity;
    tensor * _Nullable biases;
    tensor * _Nullable biasesVelocity;
    tensor * _Nullable activations;
    tensor * _Nullable affineTransformations;
    tensor * _Nullable costWeightDerivatives;
    tensor * _Nullable costBiasDerivatives;
    tensor * _Nullable batchCostWeightDeriv;
    tensor * _Nullable batchCostBiasDeriv;
    
    Train * _Nullable train;
    
    void (* _Nonnull kernelInitializers[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, void * _Nonnull kernel,  int l, int offset);
    float (* _Nonnull activationFunctions[MAX_NUMBER_NETWORK_LAYERS])(float z, float * _Nullable vec, unsigned int * _Nullable n);
    float (* _Nonnull activationDerivatives[MAX_NUMBER_NETWORK_LAYERS])(float z);
} dense_network;

typedef struct NeuralNetwork {
    
    data * _Nullable data;
    float * _Nullable * _Nullable batch;
    networkConstructor * _Nullable constructor;
    networkParameters * _Nullable parameters;
    int (* _Nullable load_params_from_input_file)(void * _Nonnull self, const char * _Nonnull paraFile);
    
    unsigned int network_num_layers; // Total number of layers in the network from input to output
    unsigned int num_conv2d_layers;  // Number of 2D convolutional layers
    int example_idx;
    
    dense_network * _Nullable dense;
    MetalCompute * _Nullable gpu;
    
    void (* _Nullable genesis)(void * _Nonnull self);
    void (* _Nullable finale)(void * _Nonnull self);
    void * _Nonnull (* _Nullable tensor)(void * _Nonnull self, tensor_dict tensor_dict);
    void (* _Nullable gpu_alloc)(void * _Nonnull self);
    
    float (* _Nullable l0_regularizer)(void * _Nonnull neural, int i, int j, int n, int stride);
    float (* _Nullable l1_regularizer)(void * _Nonnull neural, int i, int j, int n, int stride);
    float (* _Nullable l2_regularizer)(void * _Nonnull neural, int i, int j, int n, int stride);
    float (* _Nullable regularizer[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, int i, int j, int n, int stride);
    
    void (* _Nullable train_loop)(void * _Nonnull neural);
    
    // Function pointer to math ops
    float (* _Nullable math_ops)(float * _Nonnull vector, unsigned int n, char * _Nonnull op);
    
    // Function pointer to prediction evaluator
    void (* _Nullable eval_prediction)(void * _Nonnull self, char * _Nonnull dataSet, float * _Nonnull out, bool metal);
    float (* _Nullable eval_cost)(void * _Nonnull self, char * _Nonnull dataSet, bool binarization);
    
} NeuralNetwork;

NeuralNetwork * _Nonnull newNeuralNetwork(void);

#endif /* NeuralNetwork_h */

