//
//  NeuralNetwork.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 31/05/2017.
//

#include "NeuralNetwork.h"
#include "DenseNet.h"
#include "Conv2DNet.h"
#include "Memory.h"
#include "Regularization.h"

tensor * _Nullable propag_buffer = NULL;
tensor * _Nullable conv_input_matrix = NULL;

static void initNeuralData(void * _Nonnull self);

static void genesis(void * _Nonnull self);
static void finale(void * _Nonnull self);
static void gpu_alloc(void * _Nonnull self);

static void initNeuralData(void * _Nonnull self) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    nn->data->training = (training *)malloc(sizeof(training));
    nn->data->training->set = NULL;
    nn->data->training->reader = NULL;
    nn->data->training->m = 0;
    nn->data->training->n = 0;
    
    nn->data->test = (test *)malloc(sizeof(test));
    nn->data->test->set = NULL;
    nn->data->test->reader = NULL;
    nn->data->test->m = 0;
    nn->data->test->n = 0;
    
    nn->data->validation = (validation *)malloc(sizeof(validation));
    nn->data->validation->set = NULL;
    nn->data->validation->m = 0;
    nn->data->validation->n = 0;
}

static void new_network_common(void * _Nonnull neural) {
 
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    nn->num_activation_functions = 0;
    
    bzero(nn->dataPath, MAX_LONG_STRING_LENGTH);
    strcpy(nn->dataName, "<empty>");
    
    nn->constructor = allocateConstructor();
    
    for (int i=0; i<MAX_NUMBER_NETWORK_LAYERS; i++) {
         nn->kernelInitializers[i] = NULL;
    }
    
    nn->genesis = genesis;
    nn->finale = finale;
    nn->tensor = tensor_create;
    nn->gpu_alloc = gpu_alloc;
    
    nn->l0_regularizer = l0_regularizer;
    nn->l1_regularizer = l1_regularizer;
    nn->l2_regularizer = l2_regularizer;
    
    nn->math_ops = mathOps;
    
    nn->eval_prediction = evalPrediction;
    nn->eval_cost = evalCost;
}

//
// Root allocation routine of a dense (fully connected) neural network
//
BrainStormNet * _Nonnull new_dense_net(void) {
    
    BrainStormNet *nn = (BrainStormNet *)malloc(sizeof(BrainStormNet));
    *nn = (BrainStormNet){.network_num_layers=0, .is_dense_network=true, .is_conv2d_network=false};
    
    create_dense_net((void *)nn);
    new_network_common((void *)nn);
    
    // This function is only used when loading a network from a param file
    // This mode is only available for a dense netwotk
    nn->train_loop = trainLoop;
    
    return nn;
}

//
// Root allocation routine of a convolutional neural network
//
BrainStormNet * _Nonnull new_conv2d_net(void) {
    
    BrainStormNet *nn = (BrainStormNet *)malloc(sizeof(BrainStormNet));
    *nn = (BrainStormNet){.network_num_layers=0, .is_dense_network=false, .is_conv2d_network=true};
    
    create_conv2d_net((void *)nn);
    new_network_common((void *)nn);
    return nn;
}

//
//  The network genesis
//
static void genesis(void * _Nonnull self) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    // If the network is constructed with the constructor API, check that all required parameters were defined
    if (nn->constructor->networkConstruction) {
        if (nn->network_num_layers == 0) fatal(DEFAULT_CONSOLE_WRITER, "topology not defined. Use a constructor or define it in a parameter file.");
        
        if (nn->num_activation_functions == 0) {
            for (int i=0; i<nn->network_num_layers-1; i++) {
                nn->dense->activationFunctions[i] = sigmoid;
                nn->dense->activationDerivatives[i] = sigmoidPrime;
            }
        }
        
        
        char testString[MAX_LONG_STRING_LENGTH];
        bzero(testString, MAX_LONG_STRING_LENGTH);
        if (strcmp(nn->dataPath, testString) == 0) fatal(DEFAULT_CONSOLE_WRITER, "training data not defined. Use a constructor or define it in a parameter file.");
    }
    
    fprintf(stdout, "%s: create the network internal structure...\n", DEFAULT_CONSOLE_WRITER);
    fprintf(stdout, "%s: full connected network with %d layers.\n", DEFAULT_CONSOLE_WRITER, nn->network_num_layers);
    
    nn->example_idx = 0;
    
    nn->data = (data *)malloc(sizeof(data));
    nn->data->init = initNeuralData;
    nn->data->load = loadData;
    
    if (nn->is_dense_network) {
        dense_net_genesis(self);
    } else if (nn->is_conv2d_network) {
        conv2d_net_genesis(self);
        
        // ----------------------------------
        // ------- Global propagation buffer
        // ----------------------------------
        int size = 0;
        for (int l=1; l<nn->network_num_layers; l++) {
            int m = 0;
            if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION || nn->conv2d->parameters->topology[l][0] == POOLING) {
                m = nn->conv2d->parameters->topology[l][1] * nn->conv2d->parameters->topology[l][2] *
                nn->conv2d->parameters->topology[l][3];
            } else {
                m = nn->conv2d->parameters->topology[l][1];
            }
            size = max(size, m);
        }
        tensor_dict *dict = init_tensor_dict();
        dict->rank = 1;
        dict->shape[0][0][0] = size;
        propag_buffer = (tensor *)nn->tensor(self, *dict);
        memset(propag_buffer->val, 0.0f, propag_buffer->shape[0][0][0]*sizeof(float));
        free(dict);
        
        // ---------------------------------------------------------------------------------------------
        // ------- Global buffer for the input matrix used during the convolution matrix-matrix product
        // ---------------------------------------------------------------------------------------------
        int max[2] = {-INT_MAX, -INT_MAX};
        for (int l=0; l<nn->network_num_layers; l++) {
            if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
                if (nn->conv2d->parameters->topology[l][2] * nn->conv2d->parameters->topology[l][3] > max[0]) {
                    max[0] = nn->conv2d->parameters->topology[l][2] * nn->conv2d->parameters->topology[l][3];
                }
                if (nn->conv2d->parameters->topology[l-1][1] *
                    (nn->conv2d->parameters->topology[l][4]*nn->conv2d->parameters->topology[l][5]) > max[1]) {
                    max[1] = nn->conv2d->parameters->topology[l-1][1] *
                    (nn->conv2d->parameters->topology[l][4]*nn->conv2d->parameters->topology[l][5]);
                }
            }
        }
        
        dict = init_tensor_dict();
        dict->rank = 2;
        dict->shape[0][0][0] = max[0];
        dict->shape[0][1][0] = max[1];
        dict->flattening_length = 1;
        conv_input_matrix = (tensor*)nn->tensor(self, *dict);
        free(dict);
    }
}

//
// Free-up all the memory used by the network
//
static void finale(void * _Nonnull self) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    free_fmatrix(nn->data->training->set, 0, nn->data->training->m, 0, nn->data->training->n);
    free_fmatrix(nn->data->test->set, 0, nn->data->test->m, 0, nn->data->test->n);
    if (nn->data->validation->set != NULL) free_fmatrix(nn->data->validation->set, 0, nn->data->validation->m, 0, nn->data->validation->n);
    free(nn->data->training);
    free(nn->data->test);
    free(nn->data->validation);
    free(nn->data);
    free(nn->constructor);
    
    if (nn->is_dense_network) {
        dense_net_finale(self);
    } else if (nn->is_conv2d_network) {
        conv2d_net_finale(self);
        free(propag_buffer->val);
        free(propag_buffer);
        free(conv_input_matrix->val);
        free(conv_input_matrix);
    }
    
    if (nn->gpu != NULL) {
        nn->gpu->nullify();
        free(nn->gpu);
    }
}

static void gpu_alloc(void * _Nonnull self) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    nn->gpu = metalCompute();
}
