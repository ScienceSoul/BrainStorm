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

typedef struct trainer {
    gradient_descent_optimizer * _Nullable gradient_descent;
    momentum_optimizer * _Nullable momentum;
    ada_grad_optimizer * _Nullable ada_grad;
    rms_prop_optimizer * _Nullable rms_prop;
    adam_optimizer * _Nullable adam;
    void (* _Nullable next_batch)(void * _Nonnull neural, tensor * _Nonnull input, tensor * _Nonnull labels, unsigned int batch_size, int * _Nullable remainder, bool do_remainder);
    int (* _Nullable batch_range)(void * _Nonnull neural, unsigned int batch_size,  int * _Nullable remainder);
    void (* _Nullable progression)(void * _Nonnull neural, progress_dict progress_dict);
} trainer;

typedef struct dense_network {
    unsigned int num_dense_layers;
    
    tensor * _Nullable weights;
    tensor * _Nullable weights_velocity;
    tensor * _Nullable biases;
    tensor * _Nullable biases_velocity;
    tensor * _Nullable activations;
    tensor * _Nullable affine_transforms;
    tensor * _Nullable cost_weight_derivs;
    tensor * _Nullable cost_bias_derivs;
    tensor * _Nullable batch_cost_weight_derivs;
    tensor * _Nullable batch_cost_bias_derivs;
    
    dense_net_parameters * _Nullable parameters;
    trainer * _Nullable train;
    
    int (* _Nullable load_params_from_input_file)(void * _Nonnull self, const char * _Nonnull paraFile);
    float (* _Nonnull activation_functions[MAX_NUMBER_NETWORK_LAYERS])(float z, float * _Nullable vec, unsigned int * _Nullable n);
    float (* _Nonnull activation_derivatives[MAX_NUMBER_NETWORK_LAYERS])(float z);
} dense_network;

typedef struct conv2d_network {
    unsigned int num_conv2d_layers;
    unsigned int num_dense_layers;
    unsigned int num_pooling_layers;
    unsigned int num_max_pooling_layers;
    unsigned int num_infer_ops;
    unsigned int num_backpropag_ops;
    
    tensor * _Nullable conv_weights;
    tensor * _Nullable conv_weights_velocity;
    tensor * _Nullable conv_biases;
    tensor * _Nullable conv_biases_velocity;
    tensor * _Nullable conv_activations;
    tensor * _Nullable conv_affine_transforms;
    tensor * _Nullable conv_cost_weight_derivs;
    tensor * _Nullable conv_cost_bias_derivs;
    tensor * _Nullable conv_batch_cost_weight_derivs;
    tensor * _Nullable conv_batch_cost_bias_derivs;
    
    tensor * _Nullable dense_weights;
    tensor * _Nullable dense_weights_velocity;
    tensor * _Nullable dense_biases;
    tensor * _Nullable dense_biases_velocity;
    tensor * _Nullable dense_activations;
    tensor * _Nullable dense_affine_transforms;
    tensor * _Nullable dense_cost_weight_derivs;
    tensor * _Nullable dense_cost_bias_derivs;
    tensor * _Nullable dense_batch_cost_weight_derivs;
    tensor * _Nullable dense_batch_cost_bias_derivs;
    
    tensor * _Nullable flipped_weights;
    tensor * _Nullable kernel_matrices;
    tensor * _Nullable max_pool_indexes;
    
    tensor * _Nullable deltas_buffer;
    
    conv2d_net_parameters * _Nullable parameters;
    trainer * _Nullable train;
    
    float (* _Nonnull activation_functions[MAX_NUMBER_NETWORK_LAYERS])(float z, float * _Nullable vec, unsigned int * _Nullable n);
    float (* _Nonnull activation_derivatives[MAX_NUMBER_NETWORK_LAYERS])(float z);
    void (* _Nonnull inference_ops[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, unsigned int op, int * _Nullable advance);
    void (* _Nonnull backpropag_ops[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, unsigned int op,  int * _Nullable advance1, int * _Nullable advance2, int * _Nullable advance3);
    
    // Allocators
    void * _Nonnull (* _Nonnull conv_weights_alloc)(void * _Nonnull self, void * _Nonnull t_dict, bool reshape);
    void * _Nonnull (* _Nonnull conv_activations_alloc)(void * _Nonnull self, void *_Nonnull t_dict, bool reshape);
    void * _Nonnull (* _Nonnull conv_common_alloc)(void * _Nonnull self, void * _Nonnull t_dict, bool reshape);
    
    void * _Nonnull (* _Nonnull dense_weights_alloc)(void * _Nonnull self, void * _Nonnull t_dict, bool reshape);
    void * _Nonnull (* _Nonnull dense_common_alloc)(void * _Nonnull self, void * _Nonnull t_dict, bool reshape);
    void * _Nonnull (* _Nonnull max_pool_mask_indexes)(void * _Nonnull self, void * _Nonnull t_dict);
} conv2d_network;

typedef struct brain_storm_net {
    data * _Nullable data;
    network_constructor * _Nullable constructor;
    
    dense_network * _Nullable dense;
    conv2d_network * _Nullable conv2d;
    
    metal_compute * _Nullable gpu;

    tensor * _Nullable batch_inputs;
    tensor * _Nullable batch_labels;
    int label_step;
    
    unsigned int network_num_layers;       // Total number of layers in the network from input to output
    unsigned int num_activation_functions; // Number of activation functions used by the network
    unsigned int num_channels;
    int example_idx;
    
    bool is_dense_network;
    bool is_conv2d_network;
    bool init_biases;
    
    char data_path[MAX_LONG_STRING_LENGTH], data_name[MAX_LONG_STRING_LENGTH];
    
    activation_functions activation_functions_ref[MAX_NUMBER_NETWORK_LAYERS];
    
    void (* _Nullable genesis)(void * _Nonnull self);
    void (* _Nullable finale)(void * _Nonnull self);
    void * _Nonnull (* _Nullable tensor)(void * _Nullable self, tensor_dict tensor_dict);
    void (* _Nullable gpu_alloc)(void * _Nonnull self);
    
    
     void (* _Nonnull kernel_initializers[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull object, float * _Nullable factor, char * _Nullable mode, bool * _Nullable uniform, int layer, int offset, float * _Nullable val);
    
    float (* _Nullable l0_regularizer)(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2);
    float (* _Nullable l1_regularizer)(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2);
    float (* _Nullable l2_regularizer)(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2);
    float (* _Nullable regularizer[MAX_NUMBER_NETWORK_LAYERS])(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int offset, int stride1, int stride2);
    
    void (* _Nullable train_loop)(void * _Nonnull neural);
    
    // Function pointer to math ops
    float (* _Nullable math_ops)(float * _Nonnull vector, unsigned int n, char * _Nonnull op);
    
    // Function pointers to prediction evaluator
    void (* _Nullable eval_prediction)(void * _Nonnull self, char * _Nonnull data_set, float * _Nonnull out, bool metal);
    float (* _Nullable eval_cost)(void * _Nonnull self, char * _Nonnull data_set, bool binarization);
        
    // Function pointer to kernels (weights) flipping routine
    void (* _Nullable flip_kernels)(void * _Nonnull self);
    
    // Function pointer to deltas (errors) flipping routine
    void (* _Nullable flip_deltas)(void * _Nonnull self, unsigned int q, unsigned int fh, unsigned int fw);
    
    // Funcion pointer to routine for kernel matrices update
    void (* _Nullable kernel_mat_update)(void * _Nonnull self);
    
    //void (* _Nullable conv_mat_update)(void * _Nonnull self);
} brain_storm_net;

brain_storm_net * _Nonnull new_dense_net(void);
brain_storm_net * _Nonnull new_conv2d_net(void);

#endif /* NeuralNetwork_h */

