//
//  main.c
//  mnist
//
//  Created by Hakime Seddik on 05/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#include "BrainStorm.h"
#include "LoadMNISTDataSet.h"

void train(BrainStormNet *neural, MomentumOptimizer *optimizer) {
    
    unsigned int n_epochs = 40;
    unsigned int batch_size = 50;
    
    float **miniBatch = floatmatrix(0, batch_size-1, 0, neural->data->training->n-1);
    float out_test[neural->data->test->m];
    float out_validation[neural->data->validation->m];
    
    for (int k=1; k<=n_epochs; k++) {
        shuffle(neural->data->training->set, neural->data->training->m, neural->data->training->n);
        
        fprintf(stdout, "%s: Epoch {%d/%d}:\n", DEFAULT_CONSOLE_WRITER, k, n_epochs);
        double train_time = 0.0;
        for (int l=1; l<=neural->dense->train->batch_range((void *)neural, batch_size); l++) {
            neural->dense->train->next_batch((void *)neural, miniBatch, batch_size);
            double rt = realtime();
            optimizer->minimize((void *)neural, miniBatch, batch_size);
            rt = realtime() - rt;
            train_time += rt;
            neural->dense->train->progression((void *)neural, (progress_dict){.batch_size=batch_size,
                .percent=5});
        }
        fprintf(stdout, "%s: time to complete all training data set (s): %f\n", DEFAULT_CONSOLE_WRITER, train_time);
        
        
        neural->eval_prediction((void *)neural, "validation", out_validation, false);
        float acc_valid =  neural->math_ops(out_validation, neural->data->validation->m, "reduce sum");
        
        neural->eval_prediction((void *)neural, "test", out_test, false);
        float acc_test = neural->math_ops(out_test, neural->data->test->m, "reduce sum");
        
        fprintf(stdout, "{%d/%d}: Val accuracy: %d / Test accuracy: %d.\n", k, n_epochs, (int)acc_valid, (int)acc_test);
        fprintf(stdout, "\n");
    }
    
    free_fmatrix(miniBatch, 0, batch_size-1, 0, neural->data->training->n-1);
}

void networkFileInput(void) {
    
    // Instantiate a neural network and load its parameters
    BrainStormNet *neural = new_dense_net();
    if (neural->dense->load_params_from_input_file((void *)neural, "./parameters_mnist.dat") != 0 ) {
        fatal(DEFAULT_CONSOLE_WRITER, "failure in reading input parameters.");
    }
    
    // Create the data structures of the neural network
    neural->genesis((void *)neural);
    
    // Allocate and initialize the network data containers
    neural->data->init((void *)neural);
    
    // Provide the data readers to the network
    neural->data->training->reader = loadMnist;
    neural->data->test->reader = loadMnistTest;
    
    // Load training/test data
    neural->data->load((void *)neural, neural->dataName, neural->dataPath, "/Users/hakimeseddik/Documents/ESPRIT/MNIST/t10k-images-idx3-ubyte", true);
    
    neural->train_loop((void *)neural);
    neural->finale((void *)neural);
    free(neural);
}

void networkAPI(void) {
    
    // Instantiate a fully-connected neural network
    BrainStormNet *neural = new_dense_net();
    
    float regularization_factor=0.001f;
    
    // The feeding layer
    unsigned int shape[0];
    shape[0] = 784;
    neural->constructor->feed((void *)neural, shape, 1, NULL);
    
    // Fully connected layers
    neural->constructor->layer_dense((void *)neural,
                                     (layer_dict){.num_neurons=300, .activation=RELU, .kernel_initializer=standard_normal_initializer},
                                     &(regularizer_dict){.regularization_factor=regularization_factor, .regularizer_func=neural->l2_regularizer});
    
    neural->constructor->layer_dense((void *)neural,
                                     (layer_dict){.num_neurons=100, .activation=RELU, .kernel_initializer=standard_normal_initializer},
                                     &(regularizer_dict){.regularization_factor=regularization_factor, .regularizer_func=neural->l2_regularizer});
    
    neural->constructor->layer_dense((void *)neural,
                                     (layer_dict){.num_neurons=10, .activation=SOFTMAX, .kernel_initializer=standard_normal_initializer},
                                     &(regularizer_dict){.regularization_factor=regularization_factor, .regularizer_func=neural->l2_regularizer});
    
    neural->constructor->training_data((void *)neural, "/Users/hakimeseddik/Documents/ESPRIT/MNIST/train-images-idx3-ubyte");
    neural->constructor->split((void *)neural, 55000, 5000);
    
    int vector[10] = {0,1,2,3,4,5,6,7,8,9};
    neural->constructor->classification((void *)neural, vector, 10);
    
    // The optimizer
    MomentumOptimizer * optimizer = (MomentumOptimizer *)neural->constructor->optimizer((void *)neural, (optimizer_dict){.optimizer="momentum", .learning_rate=0.01f, .momentum=0.9f});
    
    // Create the data structures of the neural network
    neural->genesis((void *)neural);
    
    // Allocate and initialize the network data containers
    neural->data->init((void *)neural);
    
    // Provide the data readers to the network
    neural->data->training->reader = loadMnist;
    neural->data->test->reader = loadMnistTest;
    
    // Load training/test data
    neural->data->load((void *)neural, "mnist", neural->dataPath, "/Users/hakimeseddik/Documents/ESPRIT/MNIST/t10k-images-idx3-ubyte", true);
    
    // Train the network
    train(neural, optimizer);
    
    // Clean-up
    neural->finale((void *)neural);
    free(neural);
}

int main(int argc, const char * argv[]) {
    
    bool inputPutFile = false;
    if (inputPutFile) {
        networkFileInput();
    } else {
        networkAPI();
    }
    
    return 0;
}

