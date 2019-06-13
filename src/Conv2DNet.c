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
    
    brain_storm_net *nn = (brain_storm_net *)self;
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
    
    brain_storm_net *nn = (brain_storm_net *)self;
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
    
    brain_storm_net *nn = (brain_storm_net *)self;
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
    
    brain_storm_net *nn = (brain_storm_net *)self;
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
    
    brain_storm_net *nn = (brain_storm_net *)self;
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

static void * _Nonnull max_pool_mask_indexes(void * _Nonnull self, void * _Nonnull t_dict) {
    
    brain_storm_net *nn = (brain_storm_net *)self;
    tensor_dict *dict = (tensor_dict *)t_dict;
    
    int idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (nn->conv2d->parameters->topology[l][0] == POOLING && nn->conv2d->parameters->topology[l][8] == MAX_POOLING) {
            dict->shape[idx][0][0] = nn->conv2d->parameters->topology[l][1];
            dict->shape[idx][1][0] = nn->conv2d->parameters->topology[l][2];
            dict->shape[idx][2][0] = nn->conv2d->parameters->topology[l][3];
            idx++;
        }
    }
    tensor *t = NULL;
    if (idx > 0) {
        t = (tensor *)malloc(sizeof(tensor));
        t->rank = dict->rank;
        
        int tensor_length = 0;
        for (int l=0; l<idx; l++) {
            int dim = 1;
            for (int i=0; i<dict->rank; i++) {
                t->shape[l][i][0] = dict->shape[l][i][0];
                dim = dim * dict->shape[l][i][0];
            }
            tensor_length = tensor_length + dim;
        }
        t->int32_val = (int *)malloc(tensor_length*sizeof(int));
        nn->conv2d->num_max_pooling_layers = idx;
    }
    
    return (void *)t;
}

//
// Convolutional network allocation
//
void create_conv2d_net(void * _Nonnull self) {
    
    brain_storm_net *nn = (brain_storm_net *)self;
    
    nn->conv2d = (conv2d_network *)malloc(sizeof(conv2d_network));
    *(nn->conv2d) = (conv2d_network){.num_conv2d_layers=0, .num_dense_layers=0, .num_pooling_layers=0,
        .num_max_pooling_layers=0,
        .num_infer_ops=0,
        .num_backpropag_ops=0,
        .conv_weights=NULL,
        .conv_weights_velocity=NULL,
        .conv_biases=NULL,
        .conv_biases_velocity=NULL,
        .conv_activations=NULL,
        .conv_affine_transforms=NULL,
        .conv_cost_weight_derivs=NULL,
        .conv_cost_bias_derivs=NULL,
        .conv_batch_cost_weight_derivs=NULL,
        .conv_batch_cost_bias_derivs=NULL,
        .dense_weights=NULL,
        .dense_weights_velocity=NULL,
        .dense_biases=NULL,
        .dense_biases_velocity=NULL,
        .dense_activations=NULL,
        .dense_affine_transforms=NULL,
        .dense_cost_weight_derivs=NULL,
        .dense_cost_bias_derivs=NULL,
        .dense_batch_cost_weight_derivs=NULL,
        .dense_batch_cost_bias_derivs=NULL,
        .flipped_weights=NULL,
        .kernel_matrices=NULL,
        .max_pool_indexes=NULL,
        .deltas_buffer=NULL
    };
    
    nn->conv2d->train = (trainer *)malloc(sizeof(trainer));
    *(nn->conv2d->train) = (trainer){.gradient_descent=NULL, .ada_grad=NULL, .rms_prop=NULL,. adam=NULL};
    nn->conv2d->train->next_batch = next_batch;
    nn->conv2d->train->batch_range = batch_range;
    nn->conv2d->train->progression = progression;
    
    for (int i=0; i<MAX_NUMBER_NETWORK_LAYERS; i++) {
        nn->conv2d->activation_functions[i] = NULL;
        nn->conv2d->activation_derivatives[i] = NULL;
        nn->conv2d->inference_ops[i] = NULL;
        nn->conv2d->backpropag_ops[i] = NULL;
    }
    
    nn->conv2d->parameters = (conv2d_net_parameters *)malloc(sizeof(conv2d_net_parameters));
    nn->conv2d->parameters->eta = 0.0f;
    nn->conv2d->parameters->lambda = 0.0f;
    nn->conv2d->parameters->num_classifications = 0;
    nn->conv2d->parameters->max_number_nodes_in_dense_layer = 0;
    memset(nn->conv2d->parameters->topology, 0, sizeof(nn->dense->parameters->topology));
    memset(nn->conv2d->parameters->classifications, 0, sizeof(nn->dense->parameters->classifications));
    memset(nn->conv2d->parameters->split, 0, sizeof(nn->dense->parameters->split));
    
    nn->flip_kernels = flip_kernels;
    nn->flip_deltas = flip_deltas;
    nn->kernel_mat_update = kernel_mat_update;
    //nn->conv_mat_update = convMatUpdate;
    
    nn->conv2d->conv_weights_alloc = conv_weights_alloc;
    nn->conv2d->conv_activations_alloc = conv_activations_alloc;
    nn->conv2d->conv_common_alloc = conv_common_alloc;
    
    nn->conv2d->dense_weights_alloc = dense_weights_alloc;
    nn->conv2d->dense_common_alloc = dense_common_alloc;
    nn->conv2d->max_pool_mask_indexes = max_pool_mask_indexes;
}

//
// Convolutional network genesis
//
void conv2d_net_genesis(void * _Nonnull self) {
    
    brain_storm_net *nn = (brain_storm_net *)self;
    
    if (nn->conv2d->parameters->split[0] == 0 || nn->conv2d->parameters->split[1] == 0) fatal(DEFAULT_CONSOLE_WRITER, "data split not defined. Use a constructor or define it in a parameter file.");
    
    if (nn->conv2d->parameters->num_classifications == 0)  fatal(DEFAULT_CONSOLE_WRITER, "classification not defined. Use a constructor or define it in a parameter file.");
    
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
        dict->init_weights = true;
        nn->conv2d->conv_weights = (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, true);
        
        dict->init_weights = false;
        if (nn->conv2d->conv_cost_weight_derivs == NULL)
            nn->conv2d->conv_cost_weight_derivs = (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->conv_batch_cost_weight_derivs == NULL)
            nn->conv2d->conv_batch_cost_weight_derivs = (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->train->momentum != NULL) {
            if (nn->conv2d->conv_weights_velocity == NULL) {
                nn->conv2d->conv_weights_velocity = (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->ada_grad != NULL) {
            if (nn->conv2d->train->ada_grad->conv2d->cost_weight_derivative_squared_accumulated == NULL) {
                nn->conv2d->train->ada_grad->conv2d->cost_weight_derivative_squared_accumulated =
                                (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->rms_prop != NULL) {
            if (nn->conv2d->train->rms_prop->conv2d->cost_weight_derivative_squared_accumulated == NULL) {
                nn->conv2d->train->rms_prop->conv2d->cost_weight_derivative_squared_accumulated =
                                (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->adam != NULL) {
            if (nn->conv2d->train->adam->conv2d->weights_biased_first_moment_estimate == NULL) {
                nn->conv2d->train->adam->conv2d->weights_biased_first_moment_estimate =
                                (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
            }
            if (nn->conv2d->train->adam->conv2d->weights_biased_second_moment_estimate == NULL) {
                nn->conv2d->train->adam->conv2d->weights_biased_second_moment_estimate =
                                (tensor *)nn->conv2d->conv_weights_alloc(self, (void *)dict, false);
            }
        }
    }
    
    if (nn->conv2d->conv_biases == NULL) {
        // Tensors for shared biases for the layer l are stored as
        // Shape[fn] where:
        // fn: is the number of feature maps at the layer l
        
        dict->rank = 1;
        dict->init_weights = false;
        nn->conv2d->conv_biases = (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, true);
        if (nn->init_biases) {
            int offset = 0;
            for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
                int step = 1;
                for (int i=0; i<nn->conv2d->conv_biases->rank; i++) {
                    step = step * nn->conv2d->conv_biases->shape[l][i][0];
                }
                random_normal_initializer(nn->conv2d->conv_biases, NULL, NULL, NULL, l, offset, NULL);
                offset = offset + step;
            }
        }
        
        if (nn->conv2d->conv_cost_bias_derivs == NULL)
            nn->conv2d->conv_cost_bias_derivs = (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->conv_batch_cost_bias_derivs == NULL)
            nn->conv2d->conv_batch_cost_bias_derivs = (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->train->momentum != NULL) {
            if (nn->conv2d->conv_biases_velocity == NULL) {
                nn->conv2d->conv_biases_velocity = (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
            }
        }
        if (nn->conv2d->train->ada_grad != NULL) {
            if (nn->conv2d->train->ada_grad->conv2d->cost_bias_derivative_squared_accumulated == NULL) {
                nn->conv2d->train->ada_grad->conv2d->cost_bias_derivative_squared_accumulated =
                                (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->rms_prop != NULL) {
            if (nn->conv2d->train->rms_prop->conv2d->cost_bias_derivative_squared_accumulated == NULL) {
                nn->conv2d->train->rms_prop->conv2d->cost_bias_derivative_squared_accumulated =
                                (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->adam != NULL) {
            if (nn->conv2d->train->adam->conv2d->biases_biased_first_moment_estimate == NULL) {
                nn->conv2d->train->adam->conv2d->biases_biased_first_moment_estimate =
                                (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, false);
            }
            if (nn->conv2d->train->adam->conv2d->biases_biased_second_moment_estimate == NULL) {
                nn->conv2d->train->adam->conv2d->biases_biased_second_moment_estimate =
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
        nn->conv2d->conv_activations = (tensor *)nn->conv2d->conv_activations_alloc(self, (void *)dict, true);
    }
    
    if (nn->conv2d->conv_affine_transforms == NULL) {
        dict->rank = 3;
        nn->conv2d->conv_affine_transforms = (tensor *)nn->conv2d->conv_common_alloc(self, (void *)dict, true);
    }
    
    // ------------------------------------------------------------------------
    // ------- The fully connected layers
    // ------------------------------------------------------------------------
    
    if (nn->conv2d->dense_weights == NULL) {
        dict->rank = 2;
        dict->init_weights = true;
        nn->conv2d->dense_weights =  (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, true);
        
        dict->init_weights = false;
        if (nn->conv2d->dense_cost_weight_derivs == NULL)
            nn->conv2d->dense_cost_weight_derivs = (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->dense_batch_cost_weight_derivs == NULL)
            nn->conv2d->dense_batch_cost_weight_derivs = (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->train->momentum != NULL) {
            if (nn->conv2d->dense_weights_velocity == NULL) {
                nn->conv2d->dense_weights_velocity = (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->ada_grad != NULL) {
            if (nn->conv2d->train->ada_grad->dense->cost_weight_derivative_squared_accumulated == NULL) {
                nn->conv2d->train->ada_grad->dense->cost_weight_derivative_squared_accumulated =
                                (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->rms_prop != NULL) {
            if (nn->conv2d->train->rms_prop->dense->cost_weight_derivative_squared_accumulated == NULL) {
                nn->conv2d->train->rms_prop->dense->cost_weight_derivative_squared_accumulated =
                                (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->adam != NULL) {
            if (nn->conv2d->train->adam->dense->weights_biased_first_moment_estimate == NULL) {
                nn->conv2d->train->adam->dense->weights_biased_first_moment_estimate =
                                (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
            }
            if (nn->conv2d->train->adam->dense->weights_biased_second_moment_estimate == NULL) {
                nn->conv2d->train->adam->dense->weights_biased_second_moment_estimate =
                                (tensor *)nn->conv2d->dense_weights_alloc(self, (void *)dict, false);
            }
        }
    }
    
    if (nn->conv2d->dense_biases == NULL) {
        dict->rank = 1;
        dict->init_weights = false;
        nn->conv2d->dense_biases = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, true);
        if (nn->init_biases) {
            int offset = 0;
            for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
                int step = 1;
                for (int i=0; i<nn->conv2d->dense_biases->rank; i++) {
                    step = step * nn->conv2d->dense_biases->shape[l][i][0];
                }
                random_normal_initializer(nn->conv2d->dense_biases, NULL, NULL, NULL, l, offset, NULL);
                offset = offset + step;
            }
        }
        
        if (nn->conv2d->dense_cost_bias_derivs == NULL)
            nn->conv2d->dense_cost_bias_derivs = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->dense_batch_cost_bias_derivs == NULL)
            nn->conv2d->dense_batch_cost_bias_derivs = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
        
        if (nn->conv2d->train->momentum != NULL) {
            if (nn->conv2d->dense_biases_velocity == NULL) {
                nn->conv2d->dense_biases_velocity = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->ada_grad != NULL) {
            if (nn->conv2d->train->ada_grad->dense->cost_bias_derivative_squared_accumulated == NULL) {
                nn->conv2d->train->ada_grad->dense->cost_bias_derivative_squared_accumulated =
                                (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
            }
            
        }
        
        if (nn->conv2d->train->rms_prop != NULL) {
            if (nn->conv2d->train->rms_prop->dense->cost_bias_derivative_squared_accumulated == NULL) {
                nn->conv2d->train->rms_prop->dense->cost_bias_derivative_squared_accumulated =
                                (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
            }
        }
        
        if (nn->conv2d->train->adam != NULL) {
            if (nn->conv2d->train->adam->dense->biases_biased_first_moment_estimate == NULL) {
                nn->conv2d->train->adam->dense->biases_biased_first_moment_estimate =
                                (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
            }
            if (nn->conv2d->train->adam->dense->biases_biased_second_moment_estimate == NULL) {
                nn->conv2d->train->adam->dense->biases_biased_second_moment_estimate =
                                (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
            }
        }
    }
    
    if (nn->conv2d->dense_activations == NULL) {
        dict->rank = 1;
        nn->conv2d->dense_activations = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, true);
        
        if (nn->conv2d->dense_affine_transforms == NULL)
            nn->conv2d->dense_affine_transforms = (tensor *)nn->conv2d->dense_common_alloc(self, (void *)dict, false);
    }
    
    // --------------------------------------------------------------------
    // ------- Storage for the deltas during backpropagation
    // --------------------------------------------------------------------
    int size = 0;
    for (int l=1; l<nn->network_num_layers; l++) {
        if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
            size = max(size, (nn->conv2d->parameters->topology[l][1] * nn->conv2d->parameters->topology[l][2] * nn->conv2d->parameters->topology[l][3]));
        }
    }
    dict->rank = 1;
    dict->shape[0][0][0] = size;
    dict->flattening_length = 1;
    dict->init_weights = false;
    nn->conv2d->deltas_buffer = (tensor *)nn->tensor(self, *dict);
    memset(nn->conv2d->deltas_buffer->val, 0.0f, size*sizeof(float));
    
    // ----------------------------------------------------
    // ------- Storage for the flipped kernels (weights)
    // ----------------------------------------------------
    dict->rank = 4;
    int idx = 0;
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
    dict->init_weights = false;
    nn->conv2d->flipped_weights = (tensor *)nn->tensor(self, *dict);
    
    // -------------------------------------------------------------------------------------------------
    // ------- The convolution operation is implemented as matrix-matrix product.
    // ------- The following allocates the kernel matrices used during the product
    // -------------------------------------------------------------------------------------------------
    dict->rank = 2;
    idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
            dict->shape[idx][0][0] = nn->conv2d->parameters->topology[l-1][1] *
                                     (nn->conv2d->parameters->topology[l][4]*nn->conv2d->parameters->topology[l][5]);
            dict->shape[idx][1][0] = nn->conv2d->parameters->topology[l][1];
            idx++;
        }
    }
    fprintf(stdout, "%s: kernel matrices dimension: \n", DEFAULT_CONSOLE_WRITER);
    fprintf(stdout, "{\n");
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        fprintf(stdout, "\t %d x %d\n", dict->shape[l][0][0], dict->shape[l][1][0]);
    }
    fprintf(stdout, "}\n");
    dict->flattening_length = nn->conv2d->num_conv2d_layers;
    nn->conv2d->kernel_matrices = (tensor*)nn->tensor(self, *dict);
    
    // If max pooling is used, allocate a mask which is used during backpropagation
    dict->rank = 3;
    nn->conv2d->max_pool_indexes = (tensor *)nn->conv2d->max_pool_mask_indexes(self, (void *)dict);
    
    // Initialize the convolution matrices with the flipped initial kernels (weights)
    nn->flip_kernels(self);
    nn->kernel_mat_update(self);
    //nn->conv_mat_update((void *)nn);
    
    free(dict);
}

//
// Convolutional network destruction
//
void conv2d_net_finale(void * _Nonnull self) {
    
    brain_storm_net *nn = (brain_storm_net *)self;
    
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
    
    if (nn->conv2d->conv_affine_transforms != NULL) {
        free(nn->conv2d->conv_affine_transforms->val);
        free(nn->conv2d->conv_affine_transforms);
    }
    
    if (nn->conv2d->conv_cost_weight_derivs != NULL) {
        free(nn->conv2d->conv_cost_weight_derivs->val);
        free(nn->conv2d->conv_cost_weight_derivs);
    }
    
    if (nn->conv2d->conv_cost_bias_derivs != NULL) {
        free(nn->conv2d->conv_cost_bias_derivs->val);
        free(nn->conv2d->conv_cost_bias_derivs);
    }
    
    if (nn->conv2d->conv_batch_cost_weight_derivs != NULL) {
        free(nn->conv2d->conv_batch_cost_weight_derivs->val);
        free(nn->conv2d->conv_batch_cost_weight_derivs);
    }
    
    if (nn->conv2d->conv_batch_cost_bias_derivs != NULL) {
        free(nn->conv2d->conv_batch_cost_bias_derivs->val);
        free(nn->conv2d->conv_batch_cost_bias_derivs);
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
    if (nn->conv2d->dense_affine_transforms != NULL) {
        free(nn->conv2d->dense_affine_transforms->val);
        free(nn->conv2d->dense_affine_transforms);
    }
    if (nn->conv2d->dense_cost_weight_derivs != NULL) {
        free(nn->conv2d->dense_cost_weight_derivs->val);
        free(nn->conv2d->dense_cost_weight_derivs);
    }
    if (nn->conv2d->dense_cost_bias_derivs != NULL) {
        free(nn->conv2d->dense_cost_bias_derivs->val);
        free(nn->conv2d->dense_cost_bias_derivs);
    }
    if (nn->conv2d->dense_batch_cost_weight_derivs != NULL) {
        free(nn->conv2d->dense_batch_cost_weight_derivs->val);
        free(nn->conv2d->dense_batch_cost_weight_derivs);
    }
    if (nn->conv2d->dense_batch_cost_bias_derivs != NULL) {
        free(nn->conv2d->dense_batch_cost_bias_derivs->val);
        free(nn->conv2d->dense_batch_cost_bias_derivs);
    }
    
    // ------------------------------------------------------------------------
    // ------- Free up the optimizer
    // ------------------------------------------------------------------------
    if (nn->conv2d->train != NULL) {
        if (nn->conv2d->train->gradient_descent != NULL) {
            free(nn->conv2d->train->gradient_descent);
        }
        
        if (nn->conv2d->train->momentum != NULL) {
            if (nn->conv2d->conv_weights_velocity != NULL) {
                free(nn->conv2d->conv_weights_velocity->val);
                free(nn->conv2d->conv_weights_velocity);
            }
            if (nn->conv2d->conv_biases_velocity != NULL) {
                free(nn->conv2d->conv_biases_velocity->val);
                free(nn->conv2d->conv_biases_velocity);
            }
            if (nn->conv2d->dense_weights_velocity != NULL) {
                free(nn->conv2d->dense_weights_velocity->val);
                free(nn->conv2d->dense_weights_velocity);
            }
            if (nn->conv2d->dense_biases_velocity != NULL) {
                free(nn->conv2d->dense_biases_velocity->val);
                free(nn->conv2d->dense_biases_velocity);
            }
            free(nn->conv2d->train->momentum);
        }
        
        if (nn->conv2d->train->ada_grad != NULL) {
            if (nn->conv2d->train->ada_grad->conv2d->cost_weight_derivative_squared_accumulated != NULL) {
                free(nn->conv2d->train->ada_grad->conv2d->cost_weight_derivative_squared_accumulated->val);
                free(nn->conv2d->train->ada_grad->conv2d->cost_weight_derivative_squared_accumulated);
            }
            if (nn->conv2d->train->ada_grad->conv2d->cost_bias_derivative_squared_accumulated != NULL) {
                free(nn->conv2d->train->ada_grad->conv2d->cost_bias_derivative_squared_accumulated->val);
                free(nn->conv2d->train->ada_grad->conv2d->cost_bias_derivative_squared_accumulated);
            }
            if (nn->conv2d->train->ada_grad->dense->cost_weight_derivative_squared_accumulated != NULL) {
                free(nn->conv2d->train->ada_grad->dense->cost_weight_derivative_squared_accumulated->val);
                free(nn->conv2d->train->ada_grad->dense->cost_weight_derivative_squared_accumulated);
            }
            if (nn->conv2d->train->ada_grad->dense->cost_bias_derivative_squared_accumulated != NULL) {
                free(nn->conv2d->train->ada_grad->dense->cost_bias_derivative_squared_accumulated->val);
                free(nn->conv2d->train->ada_grad->dense->cost_bias_derivative_squared_accumulated);
            }
            free(nn->conv2d->train->ada_grad->conv2d);
            free(nn->conv2d->train->ada_grad->dense);
            free(nn->conv2d->train->ada_grad);
        }
        
        if (nn->conv2d->train->rms_prop != NULL) {
            if (nn->conv2d->train->rms_prop->conv2d->cost_weight_derivative_squared_accumulated != NULL) {
                free(nn->conv2d->train->rms_prop->conv2d->cost_weight_derivative_squared_accumulated->val);
                free(nn->conv2d->train->rms_prop->conv2d->cost_weight_derivative_squared_accumulated);
            }
            if (nn->conv2d->train->rms_prop->conv2d->cost_bias_derivative_squared_accumulated != NULL) {
                free(nn->conv2d->train->rms_prop->conv2d->cost_bias_derivative_squared_accumulated->val);
                free(nn->conv2d->train->rms_prop->conv2d->cost_bias_derivative_squared_accumulated);
            }
            if (nn->conv2d->train->rms_prop->dense->cost_weight_derivative_squared_accumulated != NULL) {
                free(nn->conv2d->train->rms_prop->dense->cost_weight_derivative_squared_accumulated->val);
                free(nn->conv2d->train->rms_prop->dense->cost_weight_derivative_squared_accumulated);
            }
            if (nn->conv2d->train->rms_prop->dense->cost_bias_derivative_squared_accumulated != NULL) {
                free(nn->conv2d->train->rms_prop->dense->cost_bias_derivative_squared_accumulated->val);
                free(nn->conv2d->train->rms_prop->dense->cost_bias_derivative_squared_accumulated);
            }
            free(nn->conv2d->train->rms_prop->conv2d);
            free(nn->conv2d->train->rms_prop->dense);
            free(nn->conv2d->train->rms_prop);
        }
        
        if (nn->conv2d->train->adam != NULL) {
            if (nn->conv2d->train->adam->conv2d->weights_biased_first_moment_estimate != NULL) {
                free(nn->conv2d->train->adam->conv2d->weights_biased_first_moment_estimate->val);
                free(nn->conv2d->train->adam->conv2d->weights_biased_first_moment_estimate);
            }
            if (nn->conv2d->train->adam->conv2d->weights_biased_second_moment_estimate != NULL) {
                free(nn->conv2d->train->adam->conv2d->weights_biased_second_moment_estimate->val);
                free(nn->conv2d->train->adam->conv2d->weights_biased_second_moment_estimate);
            }
            if (nn->conv2d->train->adam->conv2d->biases_biased_first_moment_estimate != NULL) {
                free(nn->conv2d->train->adam->conv2d->biases_biased_first_moment_estimate->val);
                free(nn->conv2d->train->adam->conv2d->biases_biased_first_moment_estimate);
            }
            if (nn->conv2d->train->adam->conv2d->biases_biased_second_moment_estimate != NULL) {
                free(nn->conv2d->train->adam->conv2d->biases_biased_second_moment_estimate->val);
                free(nn->conv2d->train->adam->conv2d->biases_biased_second_moment_estimate);
            }
            
            if (nn->conv2d->train->adam->dense->weights_biased_first_moment_estimate != NULL) {
                free(nn->conv2d->train->adam->dense->weights_biased_first_moment_estimate->val);
                free(nn->conv2d->train->adam->dense->weights_biased_first_moment_estimate);
            }
            if (nn->conv2d->train->adam->dense->weights_biased_second_moment_estimate != NULL) {
                free(nn->conv2d->train->adam->dense->weights_biased_second_moment_estimate->val);
                free(nn->conv2d->train->adam->dense->weights_biased_second_moment_estimate);
            }
            if (nn->conv2d->train->adam->dense->biases_biased_first_moment_estimate != NULL) {
                free(nn->conv2d->train->adam->dense->biases_biased_first_moment_estimate->val);
                free(nn->conv2d->train->adam->dense->biases_biased_first_moment_estimate);
            }
            if (nn->conv2d->train->adam->dense->biases_biased_second_moment_estimate != NULL) {
                free(nn->conv2d->train->adam->dense->biases_biased_second_moment_estimate->val);
                free(nn->conv2d->train->adam->dense->biases_biased_second_moment_estimate);
            }
            free(nn->conv2d->train->adam->conv2d);
            free(nn->conv2d->train->adam->dense);
            free(nn->conv2d->train->adam);
        }
        free(nn->conv2d->train);
    }
    
    if (nn->conv2d->flipped_weights != NULL) {
        free(nn->conv2d->flipped_weights->val);
        free(nn->conv2d->flipped_weights);
    }
    
    if (nn->conv2d->kernel_matrices != NULL) {
        free(nn->conv2d->kernel_matrices->val);
        free(nn->conv2d->kernel_matrices);
    }
    
    if (nn->conv2d->max_pool_indexes != NULL) {
        free(nn->conv2d->max_pool_indexes->int32_val);
        free(nn->conv2d->max_pool_indexes);
    }
    
    if (nn->conv2d->deltas_buffer != NULL) {
        free(nn->conv2d->deltas_buffer->val);
        free(nn->conv2d->deltas_buffer);
    }
    
    if (nn->conv2d->parameters != NULL) free(nn->conv2d->parameters);
    if (nn->conv2d != NULL) free(nn->conv2d);
}
