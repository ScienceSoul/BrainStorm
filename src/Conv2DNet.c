//
//  Conv2DNet.c
//  BrainStorm
//
//  Created by Hakime Seddik on 07/08/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#include "Conv2DNet.h"
#include "NeuralNetwork.h"

static void * _Nonnull conv_weights_alloc(void * _Nonnull self, void * _Nonnull t_dict, bool reshape) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    tensor_dict *dict = (tensor_dict *)t_dict;
    
    if (reshape) {
        int idx = 0;
        for (int l=1; l<nn->network_num_layers; l++) {
            if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
                dict->shape[idx][0][0] = nn->conv2d->parameters->topology[l-1][1];
                dict->shape[idx][1][0] = nn->conv2d->parameters->topology[l][1];
                dict->shape[idx][2][0] = nn->conv2d->parameters->topology[l][4];
                dict->shape[idx][3][0] = nn->conv2d->parameters->topology[l][5];
                idx++;
            }
        }
    }
    dict->flattening_length = nn->conv2d->num_conv2d_layers;
    tensor *t = (tensor *)nn->tensor(self, *dict);
    
    return (void *)t;
}

static void * _Nonnull conv_activations_alloc(void * _Nonnull self, void * _Nonnull t_dict, bool reshape) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    tensor_dict *dict = (tensor_dict *)t_dict;
    
    if (reshape) {
        int idx = 0;
        for (int l=0; l<nn->network_num_layers; l++) {
            if (l == 0 || nn->conv2d->parameters->topology[l][0] == CONVOLUTION ||
                nn->conv2d->parameters->topology[l][0] == POOLING) {
                dict->shape[idx][0][0] = nn->conv2d->parameters->topology[l][1];
                dict->shape[idx][1][0] = nn->conv2d->parameters->topology[l][2];
                dict->shape[idx][2][0] = nn->conv2d->parameters->topology[l][3];
                idx++;
            }
        }
    }
    // Activations defined at feeding layer, convolution layers and pooling layers
    dict->flattening_length = nn->conv2d->num_conv2d_layers + nn->conv2d->num_pooling_layers + 1;
    tensor *t = (tensor *)nn->tensor(self, *dict);
    
    return (void *)t;
}

static void * _Nonnull conv_common_alloc(void * _Nonnull self, void * _Nonnull t_dict, bool reshape) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    tensor_dict *dict = (tensor_dict *)t_dict;
    
    if (dict->rank > 3) {
        fatal(DEFAULT_CONSOLE_WRITER, "conv_common_alloc() routine only allocates up to rank 3.");
    }
    
    if (reshape) {
        int idx = 0;
        for (int l=0; l<nn->network_num_layers; l++) {
            if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
                dict->shape[idx][0][0] = nn->conv2d->parameters->topology[l][1];
                if (dict->rank > 1) {
                    dict->shape[idx][1][0] = nn->conv2d->parameters->topology[l][2];
                    dict->shape[idx][2][0] = nn->conv2d->parameters->topology[l][3];
                }
                idx++;
            }
        }
    }
    dict->flattening_length = nn->conv2d->num_conv2d_layers;
    tensor *t = (tensor *)nn->tensor(self, *dict);
    
    return (void *)t;
}

static void * _Nonnull dense_weights_alloc(void * _Nonnull self, void * _Nonnull t_dict, bool reshape) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    tensor_dict *dict = (tensor_dict *)t_dict;
    
    if (reshape) {
        int idx = 0;
        for (int l=1; l<nn->network_num_layers; l++) {
            if (nn->conv2d->parameters->topology[l][0] == FULLY_CONNECTED) {
                int m = nn->conv2d->parameters->topology[l][1];
                int n;
                if (nn->conv2d->parameters->topology[l-1][0] == FEED || nn->conv2d->parameters->topology[l-1][0] == CONVOLUTION || nn->conv2d->parameters->topology[l-1][0] == POOLING) {
                    n =  nn->conv2d->parameters->topology[l-1][2]*nn->conv2d->parameters->topology[l-1][3]*nn->conv2d->parameters->topology[l-1][1];
                } else {
                    n = nn->conv2d->parameters->topology[l-1][1];
                }
                dict->shape[idx][0][0] = m;
                dict->shape[idx][1][0] = n;
                idx++;
            }
        }
    }
    dict->flattening_length = nn->conv2d->num_dense_layers;
    tensor *t = (tensor *)nn->tensor(self, *dict);
    
    return (void *)t;
}

static void * _Nonnull dense_common_alloc(void * _Nonnull self, void * _Nonnull t_dict, bool reshape) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    tensor_dict *dict = (tensor_dict *)t_dict;
    
    if (reshape) {
        int idx = 0;
        for (int l=1; l<nn->network_num_layers; l++) {
            if (nn->conv2d->parameters->topology[l][0] == FULLY_CONNECTED) {
                dict->shape[idx][0][0] = nn->conv2d->parameters->topology[l][1];
                idx++;
            }
        }
    }
    dict->flattening_length = nn->conv2d->num_dense_layers;
    tensor *t = (tensor *)nn->tensor(self, *dict);
    
    return (void *)t;
}

static void * _Nonnull max_pool_mask_alloc(void * _Nonnull self, void * _Nonnull t_dict) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    tensor_dict *dict = (tensor_dict *)t_dict;
    
    int idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (nn->conv2d->parameters->topology[l][0] == POOLING && nn->conv2d->parameters->topology[l][8] == MAX_POOLING) {
            dict->shape[idx][0][0] = nn->conv2d->parameters->topology[l-1][1];
            dict->shape[idx][1][0] = nn->conv2d->parameters->topology[l-1][2];
            dict->shape[idx][2][0] = nn->conv2d->parameters->topology[l-1][3];
            idx++;
        }
    }
    tensor *t = NULL;
    if (idx > 0) {
        nn->conv2d->num_max_pooling_layers = idx;
        dict->flattening_length = idx;
        t = (tensor *)nn->tensor(self, *dict);
    }
    
    return (void *)t;
}

//
// Convolutional network allocation
//
void create_conv2d_net(void * _Nonnull self) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    nn->conv2d = (conv2d_network *)malloc(sizeof(conv2d_network));
    *(nn->conv2d) = (conv2d_network){.num_conv2d_layers=0, .num_dense_layers=0, .num_pooling_layers=0,
        .num_max_pooling_layers=0,
        .num_infer_ops=0,
        .num_backpropag_ops=0,
        .conv_weights=NULL,
        .conv_weightsVelocity=NULL,
        .conv_biases=NULL,
        .conv_biasesVelocity=NULL,
        .conv_activations=NULL,
        .conv_affineTransformations=NULL,
        .conv_costWeightDerivatives=NULL,
        .conv_costBiasDerivatives=NULL,
        .conv_batchCostWeightDeriv=NULL,
        .conv_batchCostBiasDeriv=NULL,
        .dense_weights=NULL,
        .dense_weightsVelocity=NULL,
        .dense_biases=NULL,
        .dense_biasesVelocity=NULL,
        .dense_activations=NULL,
        .dense_affineTransformations=NULL,
        .dense_costWeightDerivatives=NULL,
        .dense_costBiasDerivatives=NULL,
        .dense_batchCostWeightDeriv=NULL,
        .dense_batchCostBiasDeriv=NULL,
        .flip_matrices=NULL,
        .flipped_weights=NULL,
        .conv_matrices=NULL,
        .max_pool_mask=NULL,
        .propag_upsampling=NULL
    };
    
    nn->conv2d->train = (Train *)malloc(sizeof(Train));
    *(nn->conv2d->train) = (Train){.gradient_descent=NULL, .ada_grad=NULL, .rms_prop=NULL,. adam=NULL};
    nn->conv2d->train->next_batch = nextBatch;
    nn->conv2d->train->batch_range = batchRange;
    nn->conv2d->train->progression = progression;
    
    for (int i=0; i<MAX_NUMBER_NETWORK_LAYERS; i++) {
        nn->conv2d->activationFunctions[i] = NULL;
        nn->conv2d->activationDerivatives[i] = NULL;
        nn->conv2d->kernelInitializers[i] = NULL;
        nn->conv2d->inferenceOps[i] = NULL;
        nn->conv2d->backpropagOps[i] = NULL;
    }
    
    nn->conv2d->parameters = (conv2d_net_parameters *)malloc(sizeof(conv2d_net_parameters));
    nn->conv2d->parameters->eta = 0.0f;
    nn->conv2d->parameters->lambda = 0.0f;
    nn->conv2d->parameters->numberOfClassifications = 0;
    nn->conv2d->parameters->max_number_nodes_in_dense_layer = 0;
    memset(nn->conv2d->parameters->topology, 0, sizeof(nn->dense->parameters->topology));
    memset(nn->conv2d->parameters->classifications, 0, sizeof(nn->dense->parameters->classifications));
    memset(nn->conv2d->parameters->split, 0, sizeof(nn->dense->parameters->split));
    
    nn->conv2d->conv_weights_alloc = conv_weights_alloc;
    nn->conv2d->conv_activations_alloc = conv_activations_alloc;
    nn->conv2d->conv_common_alloc = conv_common_alloc;
    
    nn->conv2d->dense_weights_alloc = dense_weights_alloc;
    nn->conv2d->dense_common_alloc = dense_common_alloc;
    nn->conv2d->max_pool_mask_alloc = max_pool_mask_alloc;
}

//
// Convolutional network genesis
//
void conv2d_net_genesis(void * _Nonnull self) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    if (nn->conv2d->parameters->split[0] == 0 || nn->conv2d->parameters->split[1] == 0) fatal(DEFAULT_CONSOLE_WRITER, "data split not defined. Use a constructor or define it in a parameter file.");
    
    if (nn->conv2d->parameters->numberOfClassifications == 0)  fatal(DEFAULT_CONSOLE_WRITER, "classification not defined. Use a constructor or define it in a parameter file.");
    
    // ------------------------------------------------------------------------
    // ------- The convolutuon layers
    // ------------------------------------------------------------------------
    
    tensor_dict *dict = init_tensor_dict();
    
    if (nn->conv2d->conv_weights == NULL) {
        // Tensors for shared weights for the layer l are stored as
        // Shape[fn-1,fn,fh,fw], where
        // fn-1: is the number of feature maps at the layer l-1.
        //       1 if previous layer is the input layer
        // fn: is the number of feature maps at the layer l
        // fh: is the height of the receptive field
        // fw: is the width of the receptive field
        
        dict->rank = 4;
        dict->init_neural_params = true;
        nn->conv2d->conv_weights = (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, true);
        
        dict->init_neural_params = false;
        if (nn->conv2d->conv_costWeightDerivatives == NULL)
            nn->conv2d->conv_costWeightDerivatives = (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->conv_batchCostWeightDeriv == NULL)
            nn->conv2d->conv_batchCostWeightDeriv = (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->train->momentum != NULL) {
            if (nn->conv2d->conv_weightsVelocity == NULL) {
                nn->conv2d->conv_weightsVelocity = (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->ada_grad != NULL) {
            if (nn->conv2d->train->ada_grad->conv2d->costWeightDerivativeSquaredAccumulated == NULL) {
                nn->conv2d->train->ada_grad->conv2d->costWeightDerivativeSquaredAccumulated =
                                (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->rms_prop != NULL) {
            if (nn->conv2d->train->rms_prop->conv2d->costWeightDerivativeSquaredAccumulated == NULL) {
                nn->conv2d->train->rms_prop->conv2d->costWeightDerivativeSquaredAccumulated =
                                (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->adam != NULL) {
            if (nn->conv2d->train->adam->conv2d->weightsBiasedFirstMomentEstimate == NULL) {
                nn->conv2d->train->adam->conv2d->weightsBiasedFirstMomentEstimate =
                                (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
            }
            if (nn->conv2d->train->adam->conv2d->weightsBiasedSecondMomentEstimate == NULL) {
                nn->conv2d->train->adam->conv2d->weightsBiasedSecondMomentEstimate =
                                (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
            }
        }
    }
    
    if (nn->conv2d->conv_biases == NULL) {
        // Tensors for shared biases for the layer l are stored as
        // Shape[fn] where:
        // fn: is the number of feature maps at the layer l
        
        dict->rank = 1;
        dict->init_neural_params = true;
        nn->conv2d->conv_biases = (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, true);
        
        dict->init_neural_params = false;
        if (nn->conv2d->conv_costBiasDerivatives == NULL)
            nn->conv2d->conv_costBiasDerivatives = (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->conv_batchCostBiasDeriv == NULL)
            nn->conv2d->conv_batchCostBiasDeriv = (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->train->momentum != NULL) {
            if (nn->conv2d->conv_biasesVelocity == NULL) {
                nn->conv2d->conv_biasesVelocity = (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
            }
        }
        if (nn->conv2d->train->ada_grad != NULL) {
            if (nn->conv2d->train->ada_grad->conv2d->costBiasDerivativeSquaredAccumulated == NULL) {
                nn->conv2d->train->ada_grad->conv2d->costBiasDerivativeSquaredAccumulated =
                                (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->rms_prop != NULL) {
            if (nn->conv2d->train->rms_prop->conv2d->costBiasDerivativeSquaredAccumulated == NULL) {
                nn->conv2d->train->rms_prop->conv2d->costBiasDerivativeSquaredAccumulated =
                                (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->adam != NULL) {
            if (nn->conv2d->train->adam->conv2d->biasesBiasedFirstMomentEstimate == NULL) {
                nn->conv2d->train->adam->conv2d->biasesBiasedFirstMomentEstimate =
                                (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
            }
            if (nn->conv2d->train->adam->conv2d->biasesBiasedSecondMomentEstimate == NULL) {
                nn->conv2d->train->adam->conv2d->biasesBiasedSecondMomentEstimate =
                                (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
            }
        }
    }
    
    if (nn->conv2d->conv_activations == NULL) {
        // Tensors for activations (and affine transformations) for the layer l
        // are stored as Shape[fn,fh,fw], where
        // fn: is the number of feature maps at the layer l
        // fh: is the height of the feature map
        // fw: is the width of the feature map
        
        dict->rank = 3;
        dict->init_neural_params = false;
        nn->conv2d->conv_activations = (tensor *)nn->conv2d->conv_activations_alloc(self, (void *)dict, true);
    }
    
    if (nn->conv2d->conv_affineTransformations == NULL) {
        dict->rank = 3;
        dict->init_neural_params = false;
        nn->conv2d->conv_affineTransformations = (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, true);
    }
    
    // ------------------------------------------------------------------------
    // ------- The fully connected layers
    // ------------------------------------------------------------------------
    
    if (nn->conv2d->dense_weights == NULL) {
        dict->rank = 2;
        dict->init_neural_params = true;
        nn->conv2d->dense_weights =  (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, true);
        
        dict->init_neural_params = false;
        if (nn->conv2d->dense_costWeightDerivatives == NULL)
            nn->conv2d->dense_costWeightDerivatives = (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->dense_batchCostWeightDeriv == NULL)
            nn->conv2d->dense_batchCostWeightDeriv = (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->train->momentum != NULL) {
            if (nn->conv2d->dense_weightsVelocity == NULL) {
                nn->conv2d->dense_weightsVelocity = (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->ada_grad != NULL) {
            if (nn->conv2d->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated == NULL) {
                nn->conv2d->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated =
                                (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->rms_prop != NULL) {
            if (nn->conv2d->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated == NULL) {
                nn->conv2d->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated =
                                (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->adam != NULL) {
            if (nn->conv2d->train->adam->dense->weightsBiasedFirstMomentEstimate == NULL) {
                nn->conv2d->train->adam->dense->weightsBiasedFirstMomentEstimate =
                                (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
            }
            if (nn->conv2d->train->adam->dense->weightsBiasedSecondMomentEstimate == NULL) {
                nn->conv2d->train->adam->dense->weightsBiasedSecondMomentEstimate =
                                (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
            }
        }
    }
    
    if (nn->conv2d->dense_biases == NULL) {
        dict->rank = 1;
        dict->init_neural_params = true;
        nn->conv2d->dense_biases = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, true);
        
        dict->init_neural_params = false;
        if (nn->conv2d->dense_costBiasDerivatives == NULL)
            nn->conv2d->dense_costBiasDerivatives = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->dense_batchCostBiasDeriv == NULL)
            nn->conv2d->dense_batchCostBiasDeriv = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->train->momentum != NULL) {
            if (nn->conv2d->dense_biasesVelocity == NULL) {
                nn->conv2d->dense_biasesVelocity = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->ada_grad != NULL) {
            if (nn->conv2d->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated == NULL) {
                nn->conv2d->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated =
                                (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
            }
            
        }
        
        if (nn->conv2d->train->rms_prop != NULL) {
            if (nn->conv2d->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated == NULL) {
                nn->conv2d->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated =
                                (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->adam != NULL) {
            if (nn->conv2d->train->adam->dense->biasesBiasedFirstMomentEstimate == NULL) {
                nn->conv2d->train->adam->dense->biasesBiasedFirstMomentEstimate =
                                (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
            }
            if (nn->conv2d->train->adam->dense->biasesBiasedSecondMomentEstimate == NULL) {
                nn->conv2d->train->adam->dense->biasesBiasedSecondMomentEstimate =
                                (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
            }
        }
    }
    
    if (nn->conv2d->dense_activations == NULL) {
        dict->rank = 1;
        dict->init_neural_params = false;
        nn->conv2d->dense_activations = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, true);
        
        if (nn->conv2d->dense_affineTransformations == NULL)
            nn->conv2d->dense_affineTransformations = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
    }
    
    // --------------------------------------------------------------------
    // ------- Storage for the upsampled deltas from the pooling layers
    // --------------------------------------------------------------------
    int size = 0;
    for (int l=1; l<nn->network_num_layers; l++) {
        if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
            size = max(size, (nn->conv2d->parameters->topology[l][1] * nn->conv2d->parameters->topology[l][2] * nn->conv2d->parameters->topology[l][3]));
        }
    }
    nn->conv2d->propag_upsampling = (float *)malloc(size*sizeof(float));
    memset(nn->conv2d->propag_upsampling, 0.0f, size*sizeof(float));
    
    // ------------------------------------------------------
    // ------- Matrices used to flip the kernels (weights)
    // ------------------------------------------------------
    dict->rank = 2;
    int idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
            dict->shape[idx][0][0] = nn->conv2d->parameters->topology[l][4];
            dict->shape[idx][1][0] = nn->conv2d->parameters->topology[l][5];
            idx++;
        }
    }
    dict->flattening_length = nn->conv2d->num_conv2d_layers;
    dict->init_neural_params = false;
    nn->conv2d->flip_matrices = (tensor *)nn->tensor(self, *dict);
    nn->create_flip(self);
    
    // ----------------------------------------------------
    // ------- Storage for the flipped kernels (weights)
    // ----------------------------------------------------
    dict->rank = 4;
    idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
            dict->shape[idx][0][0] = nn->conv2d->parameters->topology[l-1][1];
            dict->shape[idx][1][0] = nn->conv2d->parameters->topology[l][1];
            dict->shape[idx][2][0] = nn->conv2d->parameters->topology[l][4];
            dict->shape[idx][3][0] = nn->conv2d->parameters->topology[l][5];
            idx++;
        }
    }
    dict->flattening_length = nn->conv2d->num_conv2d_layers;
    dict->init_neural_params = false;
    nn->conv2d->flipped_weights = (tensor *)nn->tensor(self, *dict);
    
    // -------------------------------------------------------------------------------------------------
    // ------- The current implementation of the convolution operations makes use of a Matrix-Vector
    // ------- multiplication. For that purpose, a sparse matrix for each rotated convolution kernel
    // ------- at each convolution layer is created. The following allocates memory for these matrices.
    // -------------------------------------------------------------------------------------------------
    dict->rank = 4;
    idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
            dict->shape[idx][0][0] = nn->conv2d->parameters->topology[l-1][1];
            dict->shape[idx][1][0] = nn->conv2d->parameters->topology[l][1];
            dict->shape[idx][2][0] = (nn->conv2d->parameters->topology[l][2]*nn->conv2d->parameters->topology[l][3]);
            dict->shape[idx][3][0] = (nn->conv2d->parameters->topology[l-1][2]*nn->conv2d->parameters->topology[l-1][3]);
            idx++;
        }
    }
    dict->flattening_length = nn->conv2d->num_conv2d_layers;
    dict->init_neural_params = false;
    nn->conv2d->conv_matrices = (tensor *)nn->tensor(self, *dict);
    
    // If max pooling is used, allocate a mask which is used during backpropagation
    dict->rank = 3;
    nn->conv2d->max_pool_mask = nn->conv2d->max_pool_mask_alloc(self, (void *)dict);
    
    // Initialize the convolution matrices with the flipped initial kernels (weights)
    nn->flip_kernels(self);
    nn->conv_mat_update(self);
    
    free(dict);
}

//
// Convolutional network destruction
//
void conv2d_net_finale(void * _Nonnull self) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    // ------------------------------------------------------------------------
    // ------- Free up the convolutuon layers
    // ------------------------------------------------------------------------
    
    if (nn->conv2d->conv_weights != NULL) {
        free(nn->conv2d->conv_weights->val);
        free(nn->conv2d->conv_weights);
    }
    
    if (nn->conv2d->conv_biases != NULL) {
        free(nn->conv2d->conv_biases->val);
        free(nn->conv2d->conv_biases);
    }
    
    if (nn->conv2d->conv_activations != NULL) {
        free(nn->conv2d->conv_activations->val);
        free(nn->conv2d->conv_activations);
    }
    
    if (nn->conv2d->conv_affineTransformations != NULL) {
        free(nn->conv2d->conv_affineTransformations->val);
        free(nn->conv2d->conv_affineTransformations);
    }
    
    if (nn->conv2d->conv_costWeightDerivatives != NULL) {
        free(nn->conv2d->conv_costWeightDerivatives->val);
        free(nn->conv2d->conv_costWeightDerivatives);
    }
    
    if (nn->conv2d->conv_costBiasDerivatives != NULL) {
        free(nn->conv2d->conv_costBiasDerivatives->val);
        free(nn->conv2d->conv_costBiasDerivatives);
    }
    
    if (nn->conv2d->conv_batchCostWeightDeriv != NULL) {
        free(nn->conv2d->conv_batchCostWeightDeriv->val);
        free(nn->conv2d->conv_batchCostWeightDeriv);
    }
    
    if (nn->conv2d->conv_batchCostBiasDeriv != NULL) {
        free(nn->conv2d->conv_batchCostBiasDeriv->val);
        free(nn->conv2d->conv_batchCostBiasDeriv);
    }
    
    // ------------------------------------------------------------------------
    // ------- Free up the fully connected layers
    // ------------------------------------------------------------------------
    
    if (nn->conv2d->dense_weights != NULL) {
        free(nn->conv2d->dense_weights->val);
        free(nn->conv2d->dense_weights);
    }
    if (nn->conv2d->dense_biases != NULL) {
        free(nn->conv2d->dense_biases->val);
        free(nn->conv2d->dense_biases);
    }
    if (nn->conv2d->dense_activations != NULL) {
        free(nn->conv2d->dense_activations->val);
        free(nn->conv2d->dense_activations);
    }
    if (nn->conv2d->dense_affineTransformations != NULL) {
        free(nn->conv2d->dense_affineTransformations->val);
        free(nn->conv2d->dense_affineTransformations);
    }
    if (nn->conv2d->dense_costWeightDerivatives != NULL) {
        free(nn->conv2d->dense_costWeightDerivatives->val);
        free(nn->conv2d->dense_costWeightDerivatives);
    }
    if (nn->conv2d->dense_costBiasDerivatives != NULL) {
        free(nn->conv2d->dense_costBiasDerivatives->val);
        free(nn->conv2d->dense_costBiasDerivatives);
    }
    if (nn->conv2d->dense_batchCostWeightDeriv != NULL) {
        free(nn->conv2d->dense_batchCostWeightDeriv->val);
        free(nn->conv2d->dense_batchCostWeightDeriv);
    }
    if (nn->conv2d->dense_batchCostBiasDeriv != NULL) {
        free(nn->conv2d->dense_batchCostBiasDeriv->val);
        free(nn->conv2d->dense_batchCostBiasDeriv);
    }
    
    // ------------------------------------------------------------------------
    // ------- Free up the optimizer
    // ------------------------------------------------------------------------
    
    if (nn->conv2d->train->gradient_descent != NULL) {
        free(nn->conv2d->train->gradient_descent);
    }
    
    if (nn->conv2d->train->momentum != NULL) {
        if (nn->conv2d->conv_weightsVelocity != NULL) {
            free(nn->conv2d->conv_weightsVelocity->val);
            free(nn->conv2d->conv_weightsVelocity);
        }
        if (nn->conv2d->conv_biasesVelocity != NULL) {
            free(nn->conv2d->conv_biasesVelocity->val);
            free(nn->conv2d->conv_biasesVelocity);
        }
        if (nn->conv2d->dense_weightsVelocity != NULL) {
            free(nn->conv2d->dense_weightsVelocity->val);
            free(nn->conv2d->dense_weightsVelocity);
        }
        if (nn->conv2d->dense_biasesVelocity != NULL) {
            free(nn->conv2d->dense_biasesVelocity->val);
            free(nn->conv2d->dense_biasesVelocity);
        }
        free(nn->conv2d->train->momentum);
    }
    
    if (nn->conv2d->train->ada_grad != NULL) {
        if (nn->conv2d->train->ada_grad->conv2d->costWeightDerivativeSquaredAccumulated != NULL) {
            free(nn->conv2d->train->ada_grad->conv2d->costWeightDerivativeSquaredAccumulated->val);
            free(nn->conv2d->train->ada_grad->conv2d->costWeightDerivativeSquaredAccumulated);
        }
        if (nn->conv2d->train->ada_grad->conv2d->costBiasDerivativeSquaredAccumulated != NULL) {
            free(nn->conv2d->train->ada_grad->conv2d->costBiasDerivativeSquaredAccumulated->val);
            free(nn->conv2d->train->ada_grad->conv2d->costBiasDerivativeSquaredAccumulated);
        }
        if (nn->conv2d->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated != NULL) {
            free(nn->conv2d->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated->val);
            free(nn->conv2d->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated);
        }
        if (nn->conv2d->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated != NULL) {
            free(nn->conv2d->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated->val);
            free(nn->conv2d->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated);
        }
        free(nn->conv2d->train->ada_grad);
    }
    
    if (nn->conv2d->train->rms_prop != NULL) {
        if (nn->conv2d->train->rms_prop->conv2d->costWeightDerivativeSquaredAccumulated != NULL) {
            free(nn->conv2d->train->rms_prop->conv2d->costWeightDerivativeSquaredAccumulated->val);
            free(nn->conv2d->train->rms_prop->conv2d->costWeightDerivativeSquaredAccumulated);
        }
        if (nn->conv2d->train->rms_prop->conv2d->costBiasDerivativeSquaredAccumulated != NULL) {
            free(nn->conv2d->train->rms_prop->conv2d->costBiasDerivativeSquaredAccumulated->val);
            free(nn->conv2d->train->rms_prop->conv2d->costBiasDerivativeSquaredAccumulated);
        }
        if (nn->conv2d->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated != NULL) {
            free(nn->conv2d->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated->val);
            free(nn->conv2d->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated);
        }
        if (nn->conv2d->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated != NULL) {
            free(nn->conv2d->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated->val);
            free(nn->conv2d->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated);
        }
    }
    
    if (nn->conv2d->train->adam != NULL) {
        if (nn->conv2d->train->adam->conv2d->weightsBiasedFirstMomentEstimate != NULL) {
            free(nn->conv2d->train->adam->conv2d->weightsBiasedFirstMomentEstimate->val);
            free(nn->conv2d->train->adam->conv2d->weightsBiasedFirstMomentEstimate);
        }
        if (nn->conv2d->train->adam->conv2d->weightsBiasedSecondMomentEstimate != NULL) {
            free(nn->conv2d->train->adam->conv2d->weightsBiasedSecondMomentEstimate->val);
            free(nn->conv2d->train->adam->conv2d->weightsBiasedSecondMomentEstimate);
        }
        if (nn->conv2d->train->adam->conv2d->biasesBiasedFirstMomentEstimate != NULL) {
            free(nn->conv2d->train->adam->conv2d->biasesBiasedFirstMomentEstimate->val);
            free(nn->conv2d->train->adam->conv2d->biasesBiasedFirstMomentEstimate);
        }
        if (nn->conv2d->train->adam->conv2d->biasesBiasedSecondMomentEstimate != NULL) {
            free(nn->conv2d->train->adam->conv2d->biasesBiasedSecondMomentEstimate->val);
            free(nn->conv2d->train->adam->conv2d->biasesBiasedSecondMomentEstimate);
        }
        
        if (nn->conv2d->train->adam->dense->weightsBiasedFirstMomentEstimate != NULL) {
            free(nn->conv2d->train->adam->dense->weightsBiasedFirstMomentEstimate->val);
            free(nn->conv2d->train->adam->dense->weightsBiasedFirstMomentEstimate);
        }
        if (nn->conv2d->train->adam->dense->weightsBiasedSecondMomentEstimate != NULL) {
            free(nn->conv2d->train->adam->dense->weightsBiasedSecondMomentEstimate->val);
            free(nn->conv2d->train->adam->dense->weightsBiasedSecondMomentEstimate);
        }
        if (nn->conv2d->train->adam->dense->biasesBiasedFirstMomentEstimate != NULL) {
            free(nn->conv2d->train->adam->dense->biasesBiasedFirstMomentEstimate->val);
            free(nn->conv2d->train->adam->dense->biasesBiasedFirstMomentEstimate);
        }
        if (nn->conv2d->train->adam->dense->biasesBiasedSecondMomentEstimate != NULL) {
            free(nn->conv2d->train->adam->dense->biasesBiasedSecondMomentEstimate->val);
            free(nn->conv2d->train->adam->dense->biasesBiasedSecondMomentEstimate);
        }
        free(nn->conv2d->train->adam);
    }
    if (nn->conv2d->flip_matrices != NULL) {
        free(nn->conv2d->flip_matrices->val);
    }
    free(nn->conv2d->flip_matrices);
    
    if (nn->conv2d->flipped_weights != NULL) {
        free(nn->conv2d->flipped_weights->val);
    }
    free(nn->conv2d->flipped_weights);
    
    if (nn->conv2d->conv_matrices != NULL) {
        free(nn->conv2d->conv_matrices->val);
    }
    free(nn->conv2d->conv_matrices);
    
    if (nn->conv2d->max_pool_mask != NULL) {
        free(nn->conv2d->max_pool_mask->val);
    }
    free(nn->conv2d->max_pool_mask);
    
    if (nn->conv2d->propag_upsampling != NULL) free(nn->conv2d->propag_upsampling);
    if (nn->conv2d->train != NULL) free(nn->conv2d->train);
    if (nn->conv2d->parameters != NULL) free(nn->conv2d->parameters);
    if (nn->conv2d != NULL) free(nn->conv2d);
}
