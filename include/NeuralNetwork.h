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
    
    dense_net_parameters * _Nullable parameters;
    Train * _Nullable train;
    
    int (* _Nullable load_params_from_input_file)(void * _Nonnull self, const char * _Nonnull paraFile);
    void (* _Nonnull kernelInitializers[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, void * _Nonnull object,  int l, int offset);
    float (* _Nonnull activationFunctions[MAX_NUMBER_NETWORK_LAYERS])(float z, float * _Nullable vec, unsigned int * _Nullable n);
    float (* _Nonnull activationDerivatives[MAX_NUMBER_NETWORK_LAYERS])(float z);
} dense_network;

typedef struct conv2d_network {
    unsigned int num_conv2d_layers;
    unsigned int num_dense_layers;
    unsigned int num_pooling_layers;
    unsigned int num_max_pooling_layers;
    unsigned int num_infer_ops;
    unsigned int num_backpropag_ops;
    
    tensor * _Nullable conv_weights;
    tensor * _Nullable conv_weightsVelocity;
    tensor * _Nullable conv_biases;
    tensor * _Nullable conv_biasesVelocity;
    tensor * _Nullable conv_activations;
    tensor * _Nullable conv_affineTransformations;
    tensor * _Nullable conv_costWeightDerivatives;
    tensor * _Nullable conv_costBiasDerivatives;
    tensor * _Nullable conv_batchCostWeightDeriv;
    tensor * _Nullable conv_batchCostBiasDeriv;
    
    tensor * _Nullable dense_weights;
    tensor * _Nullable dense_weightsVelocity;
    tensor * _Nullable dense_biases;
    tensor * _Nullable dense_biasesVelocity;
    tensor * _Nullable dense_activations;
    tensor * _Nullable dense_affineTransformations;
    tensor * _Nullable dense_costWeightDerivatives;
    tensor * _Nullable dense_costBiasDerivatives;
    tensor * _Nullable dense_batchCostWeightDeriv;
    tensor * _Nullable dense_batchCostBiasDeriv;
    
    tensor * _Nullable flipped_weights;
    tensor * _Nullable kernel_matrices;
    tensor * _Nullable max_pool_mask;
    
    tensor * _Nullable deltas_buffer;
    
    conv2d_net_parameters * _Nullable parameters;
    Train * _Nullable train;
    
    void (* _Nonnull kernelInitializers[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, void * _Nonnull object,  int l, int offset);
    float (* _Nonnull activationFunctions[MAX_NUMBER_NETWORK_LAYERS])(float z, float * _Nullable vec, unsigned int * _Nullable n);
    float (* _Nonnull activationDerivatives[MAX_NUMBER_NETWORK_LAYERS])(float z);
    void (* _Nonnull inferenceOps[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance);
    void (* _Nonnull backpropagOps[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance1, unsigned int * _Nullable advance2, unsigned int * _Nullable advance3);
    
    // Allocators
    void * _Nonnull (* _Nonnull conv_weights_alloc)(void * _Nonnull self, void * _Nonnull t_dict, bool reshape);
    void * _Nonnull (* _Nonnull conv_activations_alloc)(void * _Nonnull self, void *_Nonnull t_dict, bool reshape);
    void * _Nonnull (* _Nonnull conv_common_alloc)(void * _Nonnull self, void * _Nonnull t_dict, bool reshape);
    
    void * _Nonnull (* _Nonnull dense_weights_alloc)(void * _Nonnull self, void * _Nonnull t_dict, bool reshape);
    void * _Nonnull (* _Nonnull dense_common_alloc)(void * _Nonnull self, void * _Nonnull t_dict, bool reshape);
    void * _Nonnull (* _Nonnull max_pool_mask_alloc)(void * _Nonnull self, void * _Nonnull t_dict);
    
} conv2d_network;

typedef struct BrainStormNet {
    
    data * _Nullable data;
    networkConstructor * _Nullable constructor;
    
    dense_network * _Nullable dense;
    conv2d_network * _Nullable conv2d;
    
    MetalCompute * _Nullable gpu;
    float * _Nullable * _Nullable batch;
    
    unsigned int network_num_layers;       // Total number of layers in the network from input to output
    unsigned int num_activation_functions; // Number of activation functions used by the network
    unsigned int num_channels;
    int example_idx;
    
    bool is_dense_network;
    bool is_conv2d_network;
    
    char dataPath[MAX_LONG_STRING_LENGTH], dataName[MAX_LONG_STRING_LENGTH];
    
    activation_functions activationFunctionsRef[MAX_NUMBER_NETWORK_LAYERS];
    
    void (* _Nullable genesis)(void * _Nonnull self);
    void (* _Nullable finale)(void * _Nonnull self);
    void * _Nonnull (* _Nullable tensor)(void * _Nonnull self, tensor_dict tensor_dict);
    void (* _Nullable gpu_alloc)(void * _Nonnull self);
    
    float (* _Nullable l0_regularizer)(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2);
    float (* _Nullable l1_regularizer)(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2);
    float (* _Nullable l2_regularizer)(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2);
    float (* _Nullable regularizer[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2);
    
    void (* _Nullable train_loop)(void * _Nonnull neural);
    
    // Function pointer to math ops
    float (* _Nullable math_ops)(float * _Nonnull vector, unsigned int n, char * _Nonnull op);
    
    // Function pointers to prediction evaluator
    void (* _Nullable eval_prediction)(void * _Nonnull self, char * _Nonnull dataSet, float * _Nonnull out, bool metal);
    float (* _Nullable eval_cost)(void * _Nonnull self, char * _Nonnull dataSet, bool binarization);
        
    // Function pointer to kernels (weights) flipping routine
    void (* _Nullable flip_kernels)(void * _Nonnull self);
    
    // Function pointer to deltas (errors) flipping routine
    void (* _Nullable flip_deltas)(void * _Nonnull self, unsigned int q, unsigned int fh, unsigned int fw);
    
    // Funcion pointer to routine for kernel matrices update
    void (* _Nullable kernel_mat_update)(void * _Nonnull self);
    
} BrainStormNet;

BrainStormNet * _Nonnull new_dense_net(void);
BrainStormNet * _Nonnull new_conv2d_net(void);

#endif /* NeuralNetwork_h */

