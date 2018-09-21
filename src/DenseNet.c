//
//  DenseNet.c
//  BrainStorm
//
//  Created by Hakime Seddik on 07/08/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#include "DenseNet.h"
#include "NeuralNetwork.h"

//
// Full-connected network allocation
//
void create_dense_net(void * _Nonnull self) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    nn->dense = (dense_network *)malloc(sizeof(dense_network));
    *(nn->dense) = (dense_network){.num_dense_layers=0,
        .weights=NULL,
        .weightsVelocity=NULL,
        .biases=NULL,
        .biasesVelocity=NULL,
        .activations=NULL,
        .affineTransformations=NULL,
        .costWeightDerivatives=NULL,
        .costBiasDerivatives=NULL,
        .batchCostWeightDeriv=NULL,
        .batchCostBiasDeriv=NULL};
    
    nn->dense->train = (Train *)malloc(sizeof(Train));
    *(nn->dense->train) = (Train){.gradient_descent=NULL, .ada_grad=NULL, .rms_prop=NULL,. adam=NULL};
    nn->dense->train->next_batch = nextBatch;
    nn->dense->train->batch_range = batchRange;
    nn->dense->train->progression = progression;
    
    for (int i=0; i<MAX_NUMBER_NETWORK_LAYERS; i++) {
        nn->dense->activationFunctions[i] = NULL;
        nn->dense->activationDerivatives[i] = NULL;
        nn->dense->kernelInitializers[i] = NULL;
    }
    
    nn->dense->parameters = (dense_net_parameters *)malloc(sizeof(dense_net_parameters));
    strcpy(nn->dense->parameters->supported_parameters[0], "data_name");
    strcpy(nn->dense->parameters->supported_parameters[1], "data");
    strcpy(nn->dense->parameters->supported_parameters[2], "topology");
    strcpy(nn->dense->parameters->supported_parameters[3], "activations");
    strcpy(nn->dense->parameters->supported_parameters[4], "split");
    strcpy(nn->dense->parameters->supported_parameters[5], "classification");
    strcpy(nn->dense->parameters->supported_parameters[6], "epochs");
    strcpy(nn->dense->parameters->supported_parameters[7], "batch_size");
    strcpy(nn->dense->parameters->supported_parameters[8], "l1_regularization");
    strcpy(nn->dense->parameters->supported_parameters[9], "l2_regularization");
    strcpy(nn->dense->parameters->supported_parameters[10], "gradient_descent_optimizer");
    strcpy(nn->dense->parameters->supported_parameters[11], "momentum_optimizer");
    strcpy(nn->dense->parameters->supported_parameters[12], "adagrad_optimizer");
    strcpy(nn->dense->parameters->supported_parameters[13], "rmsprop_optimizer");
    strcpy(nn->dense->parameters->supported_parameters[14], "adam_optimizer");
    
    nn->dense->parameters->epochs = 0;
    nn->dense->parameters->miniBatchSize = 0;
    nn->dense->parameters->eta = 0.0f;
    nn->dense->parameters->lambda = 0.0f;
    nn->dense->parameters->numberOfClassifications = 0;
    memset(nn->dense->parameters->topology, 0, sizeof(nn->dense->parameters->topology));
    memset(nn->dense->parameters->classifications, 0, sizeof(nn->dense->parameters->classifications));
    memset(nn->dense->parameters->split, 0, sizeof(nn->dense->parameters->split));
    nn->dense->load_params_from_input_file = loadParametersFromImputFile;
}

//
// Full-connected netwotk genesis
//
void dense_net_genesis(void * _Nonnull self) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    nn->dense->parameters->max_number_nodes_in_layer = max_array(nn->dense->parameters->topology, nn->network_num_layers);
    
    if (nn->dense->parameters->split[0] == 0 || nn->dense->parameters->split[1] == 0) fatal(DEFAULT_CONSOLE_WRITER, "data split not defined. Use a constructor or define it in a parameter file.");
    
    if (nn->dense->parameters->numberOfClassifications == 0) fatal(DEFAULT_CONSOLE_WRITER, "classification not defined. Use a constructor or define it in a parameter file.");
    
    if (nn->dense->weights == NULL) {
        tensor_dict dict;
        dict.rank = 2;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            dict.shape[l][0][0] = nn->dense->parameters->topology[l+1];
            dict.shape[l][1][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init_neural_params = true;
        nn->dense->weights = (tensor *)nn->tensor(self, dict);
        
        dict.init_neural_params = false;
        if (nn->dense->costWeightDerivatives == NULL)
            nn->dense->costWeightDerivatives = (tensor *)nn->tensor(self, dict);
        
        if (nn->dense->batchCostWeightDeriv == NULL)
            nn->dense->batchCostWeightDeriv = (tensor *)nn->tensor(self, dict);
        
        if (nn->dense->train->momentum != NULL) {
            if (nn->dense->weightsVelocity == NULL) {
                nn->dense->weightsVelocity = (tensor *)nn->tensor(self, dict);
            }
        }
        
        if (nn->dense->train->ada_grad != NULL) {
            if (nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated == NULL) {
                nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated = (tensor *)nn->tensor(self, dict);
            }
        }
        
        if (nn->dense->train->rms_prop != NULL) {
            if (nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated == NULL) {
                nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated = (tensor *)nn->tensor(self, dict);
            }
        }
        
        if (nn->dense->train->adam != NULL) {
            if (nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate == NULL) {
                nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate = (tensor *)nn->tensor(self, dict);
            }
            if (nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate == NULL) {
                nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate = (tensor *)nn->tensor(self,  dict);
            }
        }
    }
    
    if (nn->dense->biases == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=1; l<nn->network_num_layers; l++) {
            dict.shape[l-1][0][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init_neural_params = true;
        nn->dense->biases = (tensor *)nn->tensor(self, dict);
        
        dict.init_neural_params = false;
        if (nn->dense->costBiasDerivatives == NULL)
            nn->dense->costBiasDerivatives = (tensor *)nn->tensor(self, dict);
        
        if (nn->dense->batchCostBiasDeriv == NULL)
            nn->dense->batchCostBiasDeriv = (tensor *)nn->tensor(self, dict);
        
        if (nn->dense->train->momentum != NULL) {
            if (nn->dense->biasesVelocity ==  NULL) {
                nn->dense->biasesVelocity = (tensor *)nn->tensor(self, dict);
            }
        }
        
        if (nn->dense->train->ada_grad != NULL) {
            if (nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated == NULL) {
                nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated = (tensor *)nn->tensor(self, dict);
            }
        }
        
        if (nn->dense->train->rms_prop != NULL) {
            if (nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated == NULL) {
                nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated = (tensor *)nn->tensor(self, dict);
            }
        }
        
        if (nn->dense->train->adam != NULL) {
            if (nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate == NULL) {
                nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate = (tensor *)nn->tensor(self, dict);
            }
            if (nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate == NULL) {
                nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate = (tensor *)nn->tensor(self, dict);
            }
        }
    }
    
    if (nn->dense->activations == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=0; l<nn->network_num_layers; l++) {
            dict.shape[l][0][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers;
        dict.init_neural_params = false;
        nn->dense->activations = (tensor *)nn->tensor(self, dict);
    }
    
    if (nn->dense->affineTransformations == NULL) {
        tensor_dict dict;
        dict.rank = 1;
        for (int l=1; l<nn->network_num_layers; l++) {
            dict.shape[l-1][0][0] = nn->dense->parameters->topology[l];
        }
        dict.flattening_length = nn->network_num_layers-1;
        dict.init_neural_params = false;
        nn->dense->affineTransformations = (tensor *)nn->tensor(self, dict);
    }
}

//
// Full-connected network destruction
//
void dense_net_finale(void * _Nonnull  self) {
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    if (nn->dense->weights != NULL) {
        free(nn->dense->weights->val);
        free(nn->dense->weights);
    }
    if (nn->dense->biases != NULL) {
        free(nn->dense->biases->val);
        free(nn->dense->biases);
    }
    if (nn->dense->activations != NULL) {
        free(nn->dense->activations->val);
        free(nn->dense->activations);
    }
    if (nn->dense->affineTransformations != NULL) {
        free(nn->dense->affineTransformations->val);
        free(nn->dense->affineTransformations);
    }
    if (nn->dense->costWeightDerivatives != NULL) {
        free(nn->dense->costWeightDerivatives->val);
        free(nn->dense->costWeightDerivatives);
    }
    if (nn->dense->costBiasDerivatives != NULL) {
        free(nn->dense->costBiasDerivatives->val);
        free(nn->dense->costBiasDerivatives);
    }
    if (nn->dense->batchCostWeightDeriv != NULL) {
        free(nn->dense->batchCostWeightDeriv->val);
        free(nn->dense->batchCostWeightDeriv);
    }
    if (nn->dense->batchCostBiasDeriv != NULL) {
        free(nn->dense->batchCostBiasDeriv->val);
        free(nn->dense->batchCostBiasDeriv);
    }
    
    // ------------------------------------------------------------------------
    // ------- Free up the optimizer
    // ------------------------------------------------------------------------
    
    if (nn->dense->train->gradient_descent != NULL) {
        free(nn->dense->train->gradient_descent);
    }
    
    if (nn->dense->train->momentum != NULL) {
        if (nn->dense->weightsVelocity != NULL) {
            free(nn->dense->weightsVelocity->val);
            free(nn->dense->weightsVelocity);
        }
        if (nn->dense->biasesVelocity != NULL) {
            free(nn->dense->biasesVelocity->val);
            free(nn->dense->biasesVelocity);
        }
        free(nn->dense->train->momentum);
    }
    
    if (nn->dense->train->ada_grad != NULL) {
        if (nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated != NULL) {
            free(nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated->val);
            free(nn->dense->train->ada_grad->dense->costWeightDerivativeSquaredAccumulated);
        }
        if (nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated != NULL) {
            free(nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated->val);
            free(nn->dense->train->ada_grad->dense->costBiasDerivativeSquaredAccumulated);
        }
        free(nn->dense->train->ada_grad);
    }
    if (nn->dense->train->rms_prop != NULL) {
        if (nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated != NULL) {
            free(nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated->val);
            free(nn->dense->train->rms_prop->dense->costWeightDerivativeSquaredAccumulated);
        }
        if (nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated != NULL) {
            free(nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated->val);
            free(nn->dense->train->rms_prop->dense->costBiasDerivativeSquaredAccumulated);
        }
        free(nn->dense->train->rms_prop);
    }
    if (nn->dense->train->adam != NULL) {
        if (nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate != NULL) {
            free(nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate->val);
            free(nn->dense->train->adam->dense->weightsBiasedFirstMomentEstimate);
        }
        if (nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate != NULL) {
            free(nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate->val);
            free(nn->dense->train->adam->dense->weightsBiasedSecondMomentEstimate);
        }
        
        if (nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate != NULL) {
            free(nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate->val);
            free(nn->dense->train->adam->dense->biasesBiasedFirstMomentEstimate);
        }
        if (nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate != NULL) {
            free(nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate->val);
            free(nn->dense->train->adam->dense->biasesBiasedSecondMomentEstimate);
        }
        free(nn->dense->train->adam);
    }
    if (nn->dense->train != NULL) free(nn->dense->train);
    if (nn->dense != NULL) free(nn->dense);
}

