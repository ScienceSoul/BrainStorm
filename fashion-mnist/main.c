//
//  main.c
//  fashion-mnist
//
//  Created by Hakime Seddik on 2018/11/29.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#include "BrainStorm.h"
#include "LoadFashion-MNISTDataSet.h"

void train_dense(brain_storm_net *neural, momentum_optimizer *optimizer) {
    
    unsigned int n_epochs = 40;
    unsigned int batch_size = 50;
    
    tensor_dict *dict = init_tensor_dict();
    dict->rank = 1;
    dict->shape[0][0][0] = batch_size * neural->dense->parameters->topology[0];
    tensor *features = (tensor *)neural->tensor(NULL, *dict);
    
    dict->rank = 1;
    dict->shape[0][0][0] = batch_size * neural->dense->parameters->num_classifications;
    tensor *labels = (tensor *)neural->tensor(NULL, *dict);
    
    tensor *t1 = (tensor *)neural->data->test->set;
    tensor *t2 = (tensor *)neural->data->validation->set;
    float out_test[t1->shape[0][0][0]/neural->dense->parameters->topology[0]];
    float out_validation[t2->shape[0][0][0]/neural->dense->parameters->topology[0]];
    
    for (int k=1; k<=n_epochs; k++) {
        int num_inputs = neural->dense->parameters->topology[0];
        shuffle(neural->data->training->set, neural->data->training->labels, neural->dense->parameters->num_classifications, &num_inputs);
        
        fprintf(stdout, "Fashion-MNIST: Epoch {%d/%d}:\n", k, n_epochs);
        double train_time = 0.0;
        int remainder = 0;
        for (int l=1; l<=neural->conv2d->train->batch_range((void *)neural, batch_size, &remainder); l++) {
            neural->conv2d->train->next_batch((void *)neural, features, labels, batch_size, &remainder, false);
            double rt = realtime();
            optimizer->minimize((void *)neural, features, labels,  batch_size);
            rt = realtime() - rt;
            train_time += rt;
            neural->dense->train->progression((void *)neural, (progress_dict){.batch_size=batch_size,
                .percent=5});
        }
        if (remainder != 0) {
            neural->conv2d->train->next_batch((void *)neural, features, labels, batch_size, NULL, true);
            double rt = realtime();
            optimizer->minimize((void *)neural, features, labels, batch_size);
            rt = realtime() - rt;
            train_time += rt;
        }
        fprintf(stdout, "Fashion-MNIST: training time for epoch {%d} : %f (s).\n", k, train_time);
        
        
        neural->eval_prediction((void *)neural, "validation", out_validation, false);
        float acc_valid =  neural->math_ops(out_validation, t2->shape[0][0][0]/neural->dense->parameters->topology[0], "reduce sum");
        
        neural->eval_prediction((void *)neural, "test", out_test, false);
        float acc_test = neural->math_ops(out_test, t1->shape[0][0][0]/neural->dense->parameters->topology[0], "reduce sum");
        
        fprintf(stdout, "{%d/%d}: val accuracy: %d / test accuracy: %d.\n", k, n_epochs, (int)acc_valid, (int)acc_test);
        fprintf(stdout, "\n");
    }
    
    free(features->val);
    free(features);
    
    free(labels->val);
    free(labels);
    
    free(dict);
}

void train_conv2d(brain_storm_net *neural, momentum_optimizer *optimizer) {
    
    unsigned int n_epochs = 40;
    unsigned int batch_size = 50;
    
    tensor *t1 = (tensor *)neural->data->training->set;
    
    tensor_dict *dict = init_tensor_dict();
    dict->rank = 4;
    dict->shape[0][0][0] = batch_size;
    dict->shape[0][1][0] = t1->shape[0][1][0];
    dict->shape[0][2][0] = t1->shape[0][2][0];
    dict->shape[0][3][0] = t1->shape[0][3][0];
    tensor *features = (tensor *)neural->tensor(NULL, *dict);
    
    dict->rank = 1;
    dict->shape[0][0][0] = batch_size * neural->conv2d->parameters->num_classifications;
    tensor *labels = (tensor *)neural->tensor(NULL, *dict);
    
    tensor *t2 = (tensor *)neural->data->test->set;
    tensor *t3 = (tensor *)neural->data->validation->set;
    float out_test[t2->shape[0][0][0]];
    float out_validation[t3->shape[0][0][0]];
    
    for (int k=1; k<=n_epochs; k++) {
        shuffle(neural->data->training->set, neural->data->training->labels, neural->conv2d->parameters->num_classifications, NULL);
        
        fprintf(stdout, "Fashion-MNIST: Epoch {%d/%d}:\n", k, n_epochs);
        double train_time = 0.0;
        int remainder = 0;
        for (int l=1; l<=neural->conv2d->train->batch_range((void *)neural, batch_size, &remainder); l++) {
            neural->conv2d->train->next_batch((void *)neural, features, labels, batch_size, &remainder, false);
            double rt = realtime();
            optimizer->minimize((void *)neural, features, labels, batch_size);
            rt = realtime() - rt;
            train_time += rt;
            neural->conv2d->train->progression((void *)neural, (progress_dict){.batch_size=batch_size, .percent=5});
        }
        if (remainder != 0) {
            neural->conv2d->train->next_batch((void *)neural, features, labels, batch_size, NULL, true);
            double rt = realtime();
            optimizer->minimize((void *)neural, features, labels, batch_size);
            rt = realtime() - rt;
            train_time += rt;
        }
        fprintf(stdout, "Fashion-MNIST: training time for epoch {%d} : %f (s).\n", k, train_time);
        
        neural->eval_prediction((void *)neural, "validation", out_validation, false);
        float acc_valid = neural->math_ops(out_validation, t3->shape[0][0][0], "reduce sum");
        
        neural->eval_prediction((void *)neural, "test", out_test, false);
        float acc_test = neural->math_ops(out_test, t2->shape[0][0][0], "reduce sum");
        
        fprintf(stdout, "{%d/%d}: val accuracy: %d / test accuracy: %d.\n", k, n_epochs, (int)acc_valid, (int)acc_test);
        fprintf(stdout, "\n");
    }
    
    free(features->val);
    free(features);
    
    free(labels->val);
    free(labels);
    
    free(dict);
}

void api_fully_connected_net(void) {
    
    // Instantiate a fully-connected neural network
    brain_storm_net *neural = new_dense_net();
    
    float regularization_factor = 0.001f;
    
    // The feeding layer
    neural->constructor->feed((void *)neural, (layer_dict){.shape=784, .dimension=1});
    
    // Fully hidden connected layers
    neural->constructor->layer_dense((void *)neural,
                                     (layer_dict){.num_neurons=300, .activation=RELU, .kernel_initializer=xavier_initializer},
                                     &(regularizer_dict){.regularization_factor=regularization_factor, .regularizer_func=neural->l2_regularizer});
    
    neural->constructor->layer_dense((void *)neural,
                                     (layer_dict){.num_neurons=100, .activation=RELU, .kernel_initializer=xavier_initializer},
                                     &(regularizer_dict){.regularization_factor=regularization_factor, .regularizer_func=neural->l2_regularizer});
    
    // Output layer
    neural->constructor->layer_dense((void *)neural,
                                     (layer_dict){.num_neurons=10, .activation=SOFTMAX, .kernel_initializer=xavier_initializer},
                                     &(regularizer_dict){.regularization_factor=regularization_factor, .regularizer_func=neural->l2_regularizer});
    
    neural->constructor->training_data((void *)neural, "../Data/fashion/train-images-idx3-ubyte");
    neural->constructor->split((void *)neural, 55000, 5000);
    
    int vector[10] = {0,1,2,3,4,5,6,7,8,9};
    neural->constructor->classification((void *)neural, vector, 10);
    
    // The optimizer
    momentum_optimizer *optimizer = (momentum_optimizer *)neural->constructor->optimizer((void *)neural, (optimizer_dict){.optimizer="momentum", .learning_rate=0.01f, .momentum=0.9f});
    
    // Create the data structures of the neural network
    neural->genesis((void *)neural);
    
    // Allocate and initialize the network data containers
    neural->data->init((void *)neural);
    
    // Provide the data readers to the network
    neural->data->training->reader = load_fashion_mnist;
    neural->data->test->reader = load_fashion_mnist_test;
    
    // Load training/test data
    neural->data->load((void *)neural, "fashion-mnist", neural->data_path, "../Data/fashion/t10k-images-idx3-ubyte", true, true);
    
    // Train the network
    train_dense(neural, optimizer);
    
    // Clean-up
    neural->finale((void *)neural);
    free(neural);
}

void api_convolutional_net(void) {
    
    // Instantiate a convolutional neural network
    brain_storm_net *neural = new_conv2d_net();
    
    float regularization_factor = 0.001f;
    
    // The feeding layer
    unsigned int channels = 1;
    neural->constructor->feed((void *)neural, (layer_dict){.filters=1, .dimension=2,
        .shape=28, .channels=&channels});
    
    neural->constructor->layer_conv2d((void *)neural, (layer_dict){.filters=20, .kernel_size=5, .stride=1, .padding=VALID, .activation=RELU, .kernel_initializer=xavier_initializer}, &(regularizer_dict){.regularization_factor=regularization_factor, .regularizer_func=neural->l0_regularizer});
    
    neural->constructor->layer_pool((void *)neural, (layer_dict){.filters=20, .kernel_size=2, .stride=2, .padding=VALID, .pooling_op=AVERAGE_POOLING});
    //-----
    //neural->constructor->layer_conv2d((void *)neural, (layer_dict){.filters=40, .kernel_size=5, .stride=1, .padding=VALID, .activation=RELU, .kernel_initializer=xavier_initializer}, &(regularizer_dict){.regularization_factor=regularization_factor, .regularizer_func=neural->l0_regularizer});
    
    //neural->constructor->layer_pool((void *)neural, (layer_dict){.filters=40, .kernel_size=2, .stride=2, .padding=VALID, .pooling_op=AVERAGE_POOLING});
    //----
    neural->constructor->layer_dense((void *)neural, (layer_dict){.num_neurons=100, .activation=RELU, .kernel_initializer=xavier_initializer}, &(regularizer_dict){.regularization_factor=regularization_factor, .regularizer_func=neural->l0_regularizer});
    
    neural->constructor->layer_dense((void *)neural, (layer_dict){.num_neurons=10, .activation=SOFTMAX, .kernel_initializer=xavier_initializer}, &(regularizer_dict){.regularization_factor=regularization_factor, .regularizer_func=neural->l0_regularizer});
    
    neural->constructor->training_data((void *)neural,  "../Data/fashion/train-images-idx3-ubyte");
    neural->constructor->split((void *)neural, 55000, 5000);
    
    int vector[10] = {0,1,2,3,4,5,6,7,8,9};
    neural->constructor->classification((void *)neural, vector, 10);
    
    // The optimizer
    
    momentum_optimizer *optimizer = (momentum_optimizer *)neural->constructor->optimizer((void *)neural, (optimizer_dict){.optimizer="momentum", .learning_rate=0.01f, .momentum=0.9f});
    //gradient_descent_optimizer *optimizer = (gradient_descent_optimizer *)neural->constructor->optimizer((void *)neural, (optimizer_dict){.optimizer="gradient descent", .learning_rate=0.1f});
    //ada_grad_optimizer *optimizer = (ada_grad_optimizer *)neural->constructor->optimizer((void *)neural, (optimizer_dict){.optimizer="adagrad", .learning_rate=0.01f, .delta=1.0e-7});
    //rms_prop_optimizer *optimizer = (rms_prop_optimizer *)neural->constructor->optimizer((void *)neural, (optimizer_dict){.optimizer="rmsprop", .learning_rate=0.01f, .decay_rate1=0.9f, .delta=1.0e-6});
    //adam_optimizer *optimizer = (adam_optimizer *)neural->constructor->optimizer((void *)neural, (optimizer_dict){.optimizer="adam", .step_size=0.001f, .decay_rate1=0.9f, .decay_rate2=0.999f, .delta=1.0e-8});
    
    neural->genesis((void *)neural);
    
    neural->data->init((void *)neural);
    
    neural->data->training->reader = load_fashion_mnist;
    neural->data->test->reader = load_fashion_mnist_test;
    
    neural->data->load((void *)neural, "fashion-minsit", neural->data_path,  "../Data/fashion/t10k-images-idx3-ubyte", true, true);
    
    //Train the network
    train_conv2d(neural, optimizer);
    
    neural->finale((void *)neural);
    free(neural);
}

int main(int argc, const char * argv[]) {
    
    bool dense_net = false;
    bool conv_net = true;
    
    if (dense_net) {
        api_fully_connected_net();
    }
    if (conv_net) {
        api_convolutional_net();
    }
    
    return 0;
}
