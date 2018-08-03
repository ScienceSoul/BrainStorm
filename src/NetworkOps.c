//
//  NetworkOps.c
//  BrainStorm
//
//  Created by Hakime Seddik on 18/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#endif

#include "NeuralNetwork.h"
#include "TimeProfile.h"
#include "Memory.h"
#include "NetworkOps.h"

void feedforward(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    unsigned int stride1 = 0;
    unsigned int stride2 = 0;
    unsigned int stride3 = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->weights->shape[l][0][0];
        unsigned int n = nn->dense->weights->shape[l][1][0];
        
        float buffer[nn->dense->activations->shape[l+1][0][0]];
        memset(buffer, 0.0f, sizeof(buffer));
        
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n, 1.0, nn->dense->weights->val+stride1, (int)n, nn->dense->activations->val+stride3, 1, 0.0, buffer, 1);
#ifdef __APPLE__
        vDSP_vadd(buffer, 1, nn->dense->biases->val+stride2, 1, nn->dense->affineTransformations->val+stride2, 1, nn->dense->biases->shape[l][0][0]);
#else
        for (int i=0; i<nn->biasesDimensions[l].n; i++) {
            nn->dense_affineTransformations->val[stride2+i] = buffer[i] + nn->biases[stride2+i];
        }
#endif
        float *vec = NULL;
        unsigned int *vec_length = NULL;
        if (strcmp(nn->parameters->activationFunctions[l], "softmax") == 0) {
            vec = nn->dense->affineTransformations->val+stride2;
            vec_length = &(nn->dense->affineTransformations->shape[l][0][0]);
        }
        
        stride3 = stride3 + nn->dense->activations->shape[l][0][0];
        for (int i=0; i<nn->dense->activations->shape[l+1][0][0]; i++) {
            nn->dense->activations->val[stride3+i] = nn->dense->activationFunctions[l](nn->dense->affineTransformations->val[stride2+i], vec, vec_length);
        }
        
        nanToNum(nn->dense->activations->val+stride3, nn->dense->activations->shape[l+1][0][0]);
        
        stride1 = stride1 + (m * n);
        stride2 = stride2 + nn->dense->biases->shape[l][0][0];
    }
}

void backpropagation(void * _Nonnull self) {
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // Activations at the input layer
    for (int i=0; i<nn->parameters->number_of_features; i++) {
        nn->dense->activations->val[i] = nn->batch[nn->example_idx][i];
    }
    
    // Feedforward
    feedforward(nn);
    
    // ------------- Backward pass
    // At last layer
    
    // Stride to activations at last layer
    unsigned stride2 = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        stride2 = stride2 + nn->dense->activations->shape[l][0][0];
    }
    
    // Stride to affine transformations and dc/db at last layer
    unsigned stride3 = 0;
    for (int l=0; l<nn->network_num_layers-2; l++) {
        stride3 = stride3 + nn->dense->affineTransformations->shape[l][0][0];
    }
    
    float delta[nn->parameters->max_number_of_nodes_in_layer];
    float buffer[nn->parameters->max_number_of_nodes_in_layer];
    memset(delta, 0.0f, sizeof(delta));
    memset(buffer, 0.0f, sizeof(buffer));
    
    // Compute delta
    int k = (int)nn->parameters->number_of_features;
    for (int i=0; i<nn->dense->activations->shape[nn->network_num_layers-1][0][0]; i++) {
        delta[i] = nn->dense->activations->val[stride2+i] - nn->batch[nn->example_idx][k];
        k++;
    }
    
    //Stride to dc/dw at last layer
    unsigned int stride4 = 0;
    unsigned int m, n;
    for (int l=0; l<nn->network_num_layers-2; l++) {
        m = nn->dense->batchCostWeightDeriv->shape[l][0][0];
        n = nn->dense->batchCostWeightDeriv->shape[l][1][0];
        stride4 = stride4 + (m * n);
    }
    
    stride2 = stride2 - nn->dense->activations->shape[nn->network_num_layers-2][0][0];
    n = nn->dense->batchCostWeightDeriv->shape[nn->network_num_layers-2][1][0];
    for (int i=0; i<nn->dense->batchCostWeightDeriv->shape[nn->network_num_layers-2][0][0]; i++) {
        for (int j=0; j<nn->dense->batchCostWeightDeriv->shape[nn->network_num_layers-2][1][0]; j++) {
            nn->dense->batchCostWeightDeriv->val[stride4+((i*n)+j)] = nn->dense->activations->val[stride2+j] * delta[i];
        }
    }
    for (int i=0; i<nn->dense->batchCostBiasDeriv->shape[nn->network_num_layers-2][0][0]; i++) {
        nn->dense->batchCostBiasDeriv->val[stride3+i] = delta[i];
    }
    
    // The backward pass loop
    
    // Stride to weights at last layer
    unsigned int stride = 0;
    for (int l=0; l<nn->network_num_layers-2; l++) {
        m = nn->dense->weights->shape[l][0][0];
        n = nn->dense->weights->shape[l][1][0];
        stride = stride + (m * n);
    }
    
    for (int l=nn->network_num_layers-2; l>0; l--) {
        stride2 = stride2 - nn->dense->activations->shape[l-1][0][0];
        stride3 = stride3 - nn->dense->affineTransformations->shape[l-1][0][0];
        stride4 = stride4 - (nn->dense->batchCostWeightDeriv->shape[l-1][0][0]*nn->dense->batchCostWeightDeriv->shape[l-1][1][0]);
        
        float sp[nn->dense->affineTransformations->shape[l-1][0][0]];
        for (int i=0; i<nn->dense->affineTransformations->shape[l-1][0][0]; i++) {
            sp[i] = nn->dense->activationDerivatives[l-1](nn->dense->affineTransformations->val[stride3+i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->dense->weights->shape[l][0][0], (int)nn->dense->weights->shape[l][1][0], 1.0, nn->dense->weights->val+stride, (int)nn->dense->weights->shape[l][1][0], delta, 1, 0.0, buffer, 1);
        for (int i=0; i<nn->dense->affineTransformations->shape[l-1][0][0]; i++) {
            delta[i] = buffer[i] * sp[i];
        }
        // dc/dw at layer l
        m = nn->dense->batchCostWeightDeriv->shape[l-1][0][0];
        n = nn->dense->batchCostWeightDeriv->shape[l-1][1][0];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->batchCostWeightDeriv->val[stride4+((i*n)+j)] = nn->dense->activations->val[stride2+j] * delta[i];
            }
        }
        // dc/db at layer l
        for (int i=0; i<nn->dense->batchCostBiasDeriv->shape[l-1][0][0]; i++) {
            nn->dense->batchCostBiasDeriv->val[stride3+i] = delta[i];
        }
        
        stride = stride - (nn->dense->weights->shape[l-1][0][0] * nn->dense->weights->shape[l-1][1][0]);
    }
}

void batchAccumulation(void * _Nonnull self) {
    
    // Accumulate dcdw and dc/db
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    unsigned stride1 = 0;
    unsigned stride2 = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        unsigned int m = nn->dense->costWeightDerivatives->shape[l][0][0];
        unsigned int n = nn->dense->costWeightDerivatives->shape[l][1][0];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->dense->costWeightDerivatives->val[stride1+((i*n)+j)] = nn->dense->costWeightDerivatives->val[stride1+((i*n)+j)] + nn->dense->batchCostWeightDeriv->val[stride1+((i*n)+j)];
            }
        }
        for (int i=0; i<m; i++) {
            nn->dense->costBiasDerivatives->val[stride2+i] = nn->dense->costBiasDerivatives->val[stride2+i] + nn->dense->batchCostBiasDeriv->val[stride2+i];
        }

        stride1 = stride1 + (m * n);
        stride2 = stride2 + m;
    }
}

void miniBatchLoop(void * _Nonnull neural, unsigned int batch_size) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
#ifdef VERBOSE
    double rt = realtime();
#endif
    
    for (int i=0; i<batch_size; i++) {
        nn->example_idx = i;
        backpropagation((void *)nn);
        batchAccumulation((void *)nn);
    }
    
#ifdef VERBOSE
    rt = realtime() - rt;
    fprintf(stdout, "%s: time to complete a single mini-batch (s): %f\n", DEFAULT_CONSOLE_WRITER, rt);
#endif
}

void nextBatch(void * _Nonnull neural, float * _Nonnull * _Nonnull placeholder, unsigned int batchSize) {
    
    static int delta = 0;
    static int count = 1;
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    memcpy(*placeholder, *nn->data->training->set+delta, (batchSize*(int)nn->data->training->n)*sizeof(float));
    if (count == (int)ceil((int)nn->data->training->m/batchSize)) {
        delta = 0;
        count = 1;
    } else {
        delta = delta + (batchSize*(int)nn->data->training->n);
        count++;
    }
}

int batchRange(void * _Nonnull neural, unsigned int batchSize) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    return (int)ceil((int)nn->data->training->m/batchSize);
}

void progression(void * _Nonnull neural, progress_dict progress_dict) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    int train_size = (int)nn->data->training->m/progress_dict.batch_size;
    int step = train_size / (100/progress_dict.percent);
    static int nextPrint = 0;
    static int i = 0;
    static int count = 1;
    
    if (count == 1) nextPrint = step;
    
    i++;
    if (i >= nextPrint) {
        int percent = (100 * i) / train_size;
        fprintf(stdout, "...%d%%\n", percent);
        fflush(stdout);
        nextPrint += step;
    }
    
    if (count == (int)ceil((int)nn->data->training->m/progress_dict.batch_size)) {
        i = 0;
        nextPrint = step;
        count = 1;
    } else count++;
}

float mathOps(float * _Nonnull vector, unsigned int n, char * _Nonnull op) {
    
    float result = 0.0f;
    
    if (strcmp(op, "reduce mean") == 0) {
        vDSP_meanv(vector, 1, &result, n);
    } else if (strcmp(op, "reduce sum") == 0) {
        vDSP_sve(vector, 1, &result, n);
    } else if (strcmp(op, "reduce max") == 0) {
        vDSP_maxv(vector, 1, &result, n);
    } else if (strcmp(op, "reduce min") == 0) {
        vDSP_minv(vector, 1, &result, n);
    } else fatal(DEFAULT_CONSOLE_WRITER, "unrecognized math operation.");
    
    return result;
}

static void eval(void * _Nonnull self, float * _Nonnull * _Nonnull data, unsigned int data_size, float * _Nonnull out) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    for (int k=0; k<data_size; k++) {
        
        for (int i=0; i<nn->parameters->number_of_features; i++) {
            nn->dense->activations->val[i] = data[k][i];
        }
        
        feedforward(self);
        
        // Stride to activations at last layer
        unsigned stride = 0;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            stride = stride + nn->dense->activations->shape[l][0][0];
        }
        
        out[k] = (float)argmax(nn->dense->activations->val+stride, nn->dense->activations->shape[nn->network_num_layers-1][0][0]) == data[k][nn->parameters->number_of_features];
    }
}

void evalPrediction(void * _Nonnull self, char * _Nonnull dataSet, float * _Nonnull out, bool metal) {
    
    static bool test_check = false;
    static bool validation_check = false;
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    float **data = NULL;
    unsigned int data_size = 0;
    if (strcmp(dataSet, "validation") == 0) {
        if (!validation_check) {
            if (nn->data->validation->set == NULL) fatal(DEFAULT_CONSOLE_WRITER, "trying to evaluate prediction on validation data but the data do not exist.");
            validation_check = true;
        }
        data = nn->data->validation->set;
        data_size = nn->data->validation->m;
    } else if (strcmp(dataSet, "test") == 0) {
        if (!test_check) {
            if (nn->data->test->set == NULL) fatal(DEFAULT_CONSOLE_WRITER, "trying to evaluate prediction on test data but the data  do not exist.");
            test_check = true;
        }
        data = nn->data->test->set;
        data_size = nn->data->test->m;
    } else fatal(DEFAULT_CONSOLE_WRITER, "unrecognized data set in prediction evaluation.");
    
#ifdef __APPLE__
    if (metal) {
        unsigned int weightsTableSize = 0;
        unsigned int biasesTableSize = 0;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            weightsTableSize = weightsTableSize + (nn->dense->weights->shape[l][0][0] * nn->dense->weights->shape[l][1][0]);
            biasesTableSize = biasesTableSize + nn->dense->biases->shape[l][0][0];
        }
        
        nn->gpu->allocate_buffers((void *)nn);
        nn->gpu->prepare("feedforward");
        nn->gpu->format_data(data, data_size, nn->parameters->number_of_features);
        nn->gpu->feedforward((void *)nn, out);
        
    } else {
        eval(self, data, data_size, out);
    }
#else
    eval(self, data, data_size, out);
#endif
}

//
//  Compute the total cost function using a cross-entropy formulation
//
float evalCost(void * _Nonnull self, char * _Nonnull dataSet, bool binarization) {
    
    static bool test_check = false;
    static bool validation_check = false;
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    float **data = NULL;
    unsigned int data_size = 0;
    if (strcmp(dataSet, "training") == 0) {
        data = nn->data->training->set;
        data_size = nn->data->training->m;
    } else if (strcmp(dataSet, "validation") == 0) {
        if (!validation_check) {
            if (nn->data->validation->set == NULL) fatal(DEFAULT_CONSOLE_WRITER, "trying to evaluate cost on validation data but the data do not exist.");
            validation_check = true;
        }
        data = nn->data->validation->set;
        data_size = nn->data->validation->m;
    } else if (strcmp(dataSet, "test") == 0) {
        if (!test_check) {
            if (nn->data->test->set == NULL) fatal(DEFAULT_CONSOLE_WRITER, "trying to evaluate cost on test data but the data  do not exist.");
            test_check = true;
        }
        data = nn->data->test->set;
        data_size = nn->data->test->m;
    } else fatal(DEFAULT_CONSOLE_WRITER, "unrecognized data set in cost evaluation.");
    
    float norm, sum;
    
    float cost = 0.0f;
    for (int i=0; i<data_size; i++) {
        
        for (int j=0; j<nn->parameters->number_of_features; j++) {
            nn->dense->activations->val[j] = data[i][j];
        }
        
        feedforward(self);
        
        // Stride to activations at last layer
        unsigned stride1 = 0;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            stride1 = stride1 + nn->dense->activations->shape[l][0][0];
        }
        
        float y[nn->dense->activations->shape[nn->network_num_layers-1][0][0]];
        memset(y, 0.0f, sizeof(y));
        if (binarization == true) {
            for (int j=0; j<nn->dense->activations->shape[nn->network_num_layers-1][0][0]; j++) {
                if (data[i][nn->parameters->number_of_features] == nn->parameters->classifications[j]) {
                    y[j] = 1.0f;
                }
            }
        } else {
            int idx = (int)nn->parameters->number_of_features;
            for (int j=0; j<nn->dense->activations->shape[nn->network_num_layers-1][0][0]; j++) {
                y[j] = data[i][idx];
                idx++;
            }
        }
        cost = cost + crossEntropyCost(nn->dense->activations->val+stride1, y, nn->dense->activations->shape[nn->network_num_layers-1][0][0]) / data_size;
        
        sum = 0.0f;
        unsigned int stride = 0;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            unsigned int m = nn->dense->weights->shape[l][0][0];
            unsigned int n = nn->dense->weights->shape[l][1][0];
            norm = frobeniusNorm(nn->dense->weights->val+stride, (m * n));
            sum = sum + (norm*norm);
            stride = stride + (m * n);
        }
        cost = cost + 0.5f*(nn->parameters->lambda/(float)data_size)*sum;
    }
    
    return cost;
}

void trainLoop(void * _Nonnull  neural) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    float **miniBatch = floatmatrix(0, nn->parameters->miniBatchSize-1, 0, nn->data->training->n-1);
    float out_test[nn->data->test->m];
    
    for (int k=1; k<=nn->parameters->epochs; k++) {
        shuffle(nn->data->training->set, nn->data->training->m, nn->data->training->n);
        
        for (int l=1; l<=nn->dense->train->batch_range((void *)neural,  nn->parameters->miniBatchSize); l++) {
            nn->dense->train->next_batch((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            
            if (nn->dense->train->gradient_descent != NULL) {
                nn->dense->train->gradient_descent->minimize((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            } else if (nn->dense->train->momentum != NULL) {
                nn->dense->train->momentum->minimize((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            } else if (nn->dense->train->ada_grad != NULL) {
                nn->dense->train->ada_grad->minimize((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            } else if (nn->dense->train->rms_prop != NULL) {
                nn->dense->train->rms_prop->minimize((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            }  else if (nn->dense->train->adam != NULL) {
                nn->dense->train->adam->minimize((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            }
        }
        
        fprintf(stdout, "%s: Epoch {%d/%d}: testing network with {%u} inputs:\n", DEFAULT_CONSOLE_WRITER, k, nn->parameters->epochs, nn->data->test->m);
        nn->eval_prediction((void *)nn, "test", out_test, false);
        float acc_test = nn->math_ops(out_test, nn->data->test->m, "reduce sum");
        fprintf(stdout, "{%d/%d}: Test accuracy: %d/%d.\n", k, nn->parameters->epochs, (int)acc_test, nn->data->test->m);
        fprintf(stdout, "\n");
    }
    
    free_fmatrix(miniBatch, 0, nn->parameters->miniBatchSize-1, 0, nn->data->training->n-1);
}
