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
    
    activationNode *aNodePt = nn->networkActivations;
    affineTransformationNode *zNodePt = nn->networkAffineTransformations;
    
    unsigned int stride1 = 0;
    unsigned int stride2 = 0;
    for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
        unsigned int m = nn->weightsDimensions[l].m;
        unsigned int n = nn->weightsDimensions[l].n;
        
        aNodePt = aNodePt->next;
        zNodePt = zNodePt->next;
        float buffer[aNodePt->n];
        memset(buffer, 0.0f, sizeof(buffer));
        
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n, 1.0, nn->weights+stride1, (int)n, aNodePt->previous->a, 1, 0.0, buffer, 1);
#ifdef __APPLE__
        vDSP_vadd(buffer, 1, nn->biases+stride2, 1, zNodePt->z, 1,nn->biasesDimensions[l].n);
#else
        for (int i=0; i<nn->biasesDimensions[l].n; i++) {
            zNodePt->z[i] = buffer[i] + nn->biases[stride2+i];
        }
#endif
        float *vec = NULL;
        unsigned int *vec_length = NULL;
        if (strcmp(nn->parameters->activationFunctions[l], "softmax") == 0) {
            vec = zNodePt->z;
            vec_length = &zNodePt->n;
        }
        for (int i=0; i<aNodePt->n; i++) {
            aNodePt->a[i] = nn->activationFunctions[l](zNodePt->z[i],vec, vec_length);
        }
        nanToNum(aNodePt->a, aNodePt->n);
        
        stride1 = stride1 + (m * n);
        stride2 = stride2 + nn->biasesDimensions[l].n;
    }
}

void backpropagation(void * _Nonnull self) {
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // Activations at the input layer
    activationNode *aNodePt = nn->networkActivations;
    for (int i=0; i<nn->number_of_features; i++) {
        aNodePt->a[i] = nn->batch[nn->example_idx][i];
    }
    
    // Feedforward
    feedforward(nn);
    
    // ------------- Backward pass
    // At last layer
    
    activationNode *aTail = nn->networkActivations;
    while (aTail != NULL && aTail->next != NULL) {
        aTail = aTail->next;
    }
    affineTransformationNode *zTail = nn->networkAffineTransformations;
    while (zTail != NULL && zTail->next != NULL) {
        zTail = zTail->next;
    }
    
    float delta[nn->max_number_of_nodes_in_layer];
    float buffer[nn->max_number_of_nodes_in_layer];
    memset(delta, 0.0f, sizeof(delta));
    memset(buffer, 0.0f, sizeof(buffer));
    
    // Compute delta
    int k = (int)nn->number_of_features;
    for (int i=0; i<aTail->n; i++) {
        delta[i] = aTail->a[i] - nn->batch[nn->example_idx][k];
        k++;
    }
    
    //dc/dw and dc/db at last layer
    costWeightDerivativeNode *dcdwTail = nn->deltaNetworkCostWeightDerivatives;
    while (dcdwTail != NULL && dcdwTail->next != NULL) {
        dcdwTail = dcdwTail->next;
    }
    costBiaseDerivativeNode *dcdbTail = nn->deltaNetworkCostBiaseDerivatives;
    while (dcdbTail != NULL && dcdbTail->next != NULL) {
        dcdbTail = dcdbTail->next;
    }
    aNodePt = aTail->previous;
    for (int i=0; i<dcdwTail->m; i++) {
        for (int j=0; j<dcdwTail->n; j++) {
            dcdwTail->dcdw[i][j] = aNodePt->a[j]*delta[i];
        }
    }
    for (int i=0; i<dcdbTail->n; i++) {
        dcdbTail->dcdb[i] = delta[i];
    }
    
    // The backward pass loop
    
    // Stride to weithts at last layer
    unsigned int stride = 0;
    unsigned int m, n;
    for (int l=0; l<nn->parameters->numberOfLayers-2; l++) {
        m = nn->weightsDimensions[l].m;
        n = nn->weightsDimensions[l].n;
        stride = stride + (m * n);
    }
    
    affineTransformationNode *zNodePt = zTail->previous;
    costWeightDerivativeNode *dcdwNodePt = dcdwTail->previous;
    costBiaseDerivativeNode *dcdbNodePt = dcdbTail->previous;
    
    unsigned int l = nn->parameters->numberOfLayers - 2;
    while (dcdwNodePt != NULL && dcdbNodePt != NULL) {
        aNodePt = aNodePt->previous;
        
        float sp[zNodePt->n];
        for (int i=0; i<zNodePt->n; i++) {
            sp[i] = nn->activationDerivatives[l-1](zNodePt->z[i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->weightsDimensions[l].m, (int)nn->weightsDimensions[l].n, 1.0, nn->weights+stride, (int)nn->weightsDimensions[l].n, delta, 1, 0.0, buffer, 1);
        for (int i=0; i<zNodePt->n; i++) {
            delta[i] = buffer[i] * sp[i];
        }
        // dc/dw at layer l
        for (int i=0; i<dcdwNodePt->m; i++) {
            for (int j=0; j<dcdwNodePt->n; j++) {
                dcdwNodePt->dcdw[i][j] = aNodePt->a[j]*delta[i];
            }
        }
        // dc/db at layer l
        for (int i=0; i<dcdbNodePt->n; i++) {
            dcdbNodePt->dcdb[i] = delta[i];
        }
        
        zNodePt = zNodePt->previous;
        dcdwNodePt = dcdwNodePt->previous;
        dcdbNodePt = dcdbNodePt->previous;
        stride = stride - (nn->weightsDimensions[l-1].m * nn->weightsDimensions[l-1].n);
        l--;
    }
}

void batchAccumulation(void * _Nonnull self) {
    
    // Accumulate dcdw and dc/db
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    costWeightDerivativeNode *dcdwNodePt = nn->networkCostWeightDerivatives;
    costBiaseDerivativeNode *dcdbNodePt = nn->networkCostBiaseDerivatives;
    costWeightDerivativeNode *delta_dcdwNodePt = nn->deltaNetworkCostWeightDerivatives;
    costBiaseDerivativeNode *delta_dcdbNodePt = nn->deltaNetworkCostBiaseDerivatives;
    while (dcdwNodePt != NULL && delta_dcdwNodePt != NULL) {
        for (int i=0; i<dcdwNodePt->m; i++) {
            for (int j=0; j<dcdwNodePt->n; j++) {
                dcdwNodePt->dcdw[i][j] = dcdwNodePt->dcdw[i][j] + delta_dcdwNodePt->dcdw[i][j];
            }
        }
        for (int i=0; i<dcdbNodePt->n; i++) {
            dcdbNodePt->dcdb[i] = dcdbNodePt->dcdb[i] + delta_dcdbNodePt->dcdb[i];
        }
        dcdwNodePt = dcdwNodePt->next;
        dcdbNodePt = dcdbNodePt->next;
        delta_dcdwNodePt = delta_dcdwNodePt->next;
        delta_dcdbNodePt = delta_dcdbNodePt->next;
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
    
    activationNode *aNodePt = NULL;
    
    for (int k=0; k<data_size; k++) {
        
        aNodePt = nn->networkActivations;
        for (int i=0; i<nn->number_of_features; i++) {
            aNodePt->a[i] = data[k][i];
        }
        
        feedforward(self);
        
        aNodePt = nn->networkActivations;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
        
        out[k] = (float)argmax(aNodePt->a, aNodePt->n) == data[k][nn->number_of_features];
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
        for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
            weightsTableSize = weightsTableSize + (nn->weightsDimensions[l].m * nn->weightsDimensions[l].n);
            biasesTableSize = biasesTableSize + nn->biasesDimensions[l].n;
        }
        
        nn->gpu->allocate_buffers((void *)nn);
        nn->gpu->prepare("feedforward");
        nn->gpu->format_data(data, data_size, nn->number_of_features);
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
    activationNode *aNodePt = NULL;
    
    float cost = 0.0f;
    for (int i=0; i<data_size; i++) {
        
        aNodePt = nn->networkActivations;
        for (int j=0; j<nn->number_of_features; j++) {
            aNodePt->a[j] = data[i][j];
        }
        
        feedforward(self);
        aNodePt = nn->networkActivations;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
        
        float y[aNodePt->n];
        memset(y, 0.0f, sizeof(y));
        if (binarization == true) {
            for (int j=0; j<aNodePt->n; j++) {
                if (data[i][nn->number_of_features] == nn->parameters->classifications[j]) {
                    y[j] = 1.0f;
                }
            }
        } else {
            int idx = (int)nn->number_of_features;
            for (int j=0; j<aNodePt->n; j++) {
                y[j] = data[i][idx];
                idx++;
            }
        }
        cost = cost + crossEntropyCost(aNodePt->a, y, aNodePt->n) / data_size;
        
        sum = 0.0f;
        unsigned int stride = 0;
        for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
            unsigned int m = nn->weightsDimensions[l].m;
            unsigned int n = nn->weightsDimensions[l].n;
            norm = frobeniusNorm(nn->weights+stride, (m * n));
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
        
        for (int l=1; l<=nn->train->batch_range((void *)neural,  nn->parameters->miniBatchSize); l++) {
            nn->train->next_batch((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            
            if (nn->train->gradient_descent != NULL) {
                nn->train->gradient_descent->minimize((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            } else if (nn->train->momentum != NULL) {
                nn->train->momentum->minimize((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            } else if (nn->train->ada_grad != NULL) {
                nn->train->ada_grad->minimize((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            } else if (nn->train->rms_prop != NULL) {
                nn->train->rms_prop->minimize((void *)nn, miniBatch, nn->parameters->miniBatchSize);
            }  else if (nn->train->adam != NULL) {
                nn->train->adam->minimize((void *)nn, miniBatch, nn->parameters->miniBatchSize);
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
