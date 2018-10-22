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
#include "NetworkOps.h"
#include "DenseNetOps.h"
#include "Conv2DNetOps.c"
#include "Memory.h"

typedef void (*eval_net_type)(void * _Nonnull neural, float * _Nonnull * _Nonnull data, unsigned int data_size, float * _Nonnull out);

void miniBatchLoop(void * _Nonnull neural, unsigned int batch_size,
                   ptr_inference_func inference, ptr_backpropag_func backpropagation,
                   ptr_batch_accumul_func batch_accumulation) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
#ifdef VERBOSE
    double rt = realtime();
#endif
    
    for (int i=0; i<batch_size; i++) {
        nn->example_idx = i;
        backpropagation((void *)nn, inference);
        batch_accumulation((void *)nn);
    }
    
#ifdef VERBOSE
    rt = realtime() - rt;
    fprintf(stdout, "%s: time to complete a single mini-batch (s): %f\n", DEFAULT_CONSOLE_WRITER, rt);
#endif
}

void nextBatch(void * _Nonnull neural, float * _Nonnull * _Nonnull placeholder, unsigned int batchSize) {
    
    static int delta = 0;
    static int count = 1;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
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
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    return (int)ceil((int)nn->data->training->m/batchSize);
}

void progression(void * _Nonnull neural, progress_dict progress_dict) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
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

void eval_dense_net(void * _Nonnull neural, float * _Nonnull * _Nonnull data, unsigned int data_size, float * _Nonnull out) {
 
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Offset to activations at output layer
    unsigned int offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        offset = offset + nn->dense->activations->shape[l][0][0];
    }
    
    for (int k=0; k<data_size; k++) {
        
        for (int i=0; i<nn->num_channels; i++) {
            nn->dense->activations->val[i] = data[k][i];
        }
        inference_in_dense_net(nn);
        
        out[k] = (float)argmax(nn->dense->activations->val+offset, nn->dense->activations->shape[nn->network_num_layers-1][0][0]) == data[k][nn->num_channels];
    }
}

void eval_conv2d_net(void * _Nonnull neural, float * _Nonnull * _Nonnull data, unsigned int data_size, float * _Nonnull out) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Offset to activations at the output layer
    unsigned int offset = 0;
    for (int l=0; l<nn->conv2d->num_dense_layers-1; l++) {
        offset = offset + nn->conv2d->dense_activations->shape[l][0][0];
    }
    
    for (int k=0; k<data_size; k++) {
    
        for (int i=0; i<nn->num_channels; i++) {
            nn->conv2d->conv_activations->val[i] = data[k][i];
        }
        inference_in_conv2d_net(nn);
        
        out[k] = (float)argmax(nn->conv2d->dense_activations->val+offset, nn->conv2d->dense_activations->shape[nn->conv2d->num_dense_layers-1][0][0]) == data[k][nn->num_channels];
    }
}

static void eval(void * _Nonnull self, float * _Nonnull * _Nonnull data, unsigned int data_size, float * _Nonnull out) {
    
    static bool firstTime = true;
    static eval_net_type eval_net = NULL;
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    if (firstTime) {
        if (nn->is_dense_network) {
            eval_net = eval_dense_net;
        } else if (nn->is_conv2d_network) {
            eval_net = eval_conv2d_net;
        }
        firstTime = false;
    }
    
    eval_net(self, data, data_size, out);
}

void evalPrediction(void * _Nonnull self, char * _Nonnull dataSet, float * _Nonnull out, bool metal) {
    
    static bool test_check = false;
    static bool validation_check = false;
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
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
            if (nn->data->test->set == NULL) fatal(DEFAULT_CONSOLE_WRITER, "trying to evaluate prediction on test data but the data do not exist.");
            test_check = true;
        }
        data = nn->data->test->set;
        data_size = nn->data->test->m;
    } else fatal(DEFAULT_CONSOLE_WRITER, "unrecognized data set in prediction evaluation.");
    
#ifdef __APPLE__
    if (metal) {
        fatal(DEFAULT_CONSOLE_WRITER, "Offload evaluation to GPU broken.");
        unsigned int weightsTableSize = 0;
        unsigned int biasesTableSize = 0;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            weightsTableSize = weightsTableSize + (nn->dense->weights->shape[l][0][0] * nn->dense->weights->shape[l][1][0]);
            biasesTableSize = biasesTableSize + nn->dense->biases->shape[l][0][0];
        }
        
        nn->gpu->allocate_buffers((void *)nn);
        nn->gpu->prepare("feedforward");
        nn->gpu->format_data(data, data_size, nn->num_channels);
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
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    ptr_inference_func inference = NULL;
    if (nn->is_dense_network) {
        inference = inference_in_dense_net;
    } else if (nn->is_conv2d_network) {
        // TODO
        fatal(DEFAULT_CONSOLE_WRITER, "Cost function calculation in convolution net not implemented yet.");
    }
    
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
        
        for (int j=0; j<nn->num_channels; j++) {
            nn->dense->activations->val[j] = data[i][j];
        }
        
        inference(self);
        
        // Stride to activations at last layer
        unsigned stride1 = 0;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            stride1 = stride1 + nn->dense->activations->shape[l][0][0];
        }
        
        float y[nn->dense->activations->shape[nn->network_num_layers-1][0][0]];
        memset(y, 0.0f, sizeof(y));
        if (binarization == true) {
            for (int j=0; j<nn->dense->activations->shape[nn->network_num_layers-1][0][0]; j++) {
                if (data[i][nn->num_channels] == nn->dense->parameters->classifications[j]) {
                    y[j] = 1.0f;
                }
            }
        } else {
            int idx = (int)nn->num_channels;
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
        cost = cost + 0.5f*(nn->dense->parameters->lambda/(float)data_size)*sum;
    }
    
    return cost;
}

//
// This routine flips horizontally and vertically the
// kernels accross all convolution layers
//
void flipKernels(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    int offset_w = 0;
    int offset_f = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned int p = nn->conv2d->conv_weights->shape[l][0][0];
        unsigned int q = nn->conv2d->conv_weights->shape[l][1][0];
        unsigned int kh = nn->conv2d->conv_weights->shape[l][2][0];
        unsigned int kw = nn->conv2d->conv_weights->shape[l][3][0];
        
        int stride1_w = 0;
        for (int k=0; k<p; k++) {
            int stride2_w = 0;
            for (int ll=0; ll<q; ll++) {
                memcpy(nn->conv2d->flipped_weights->val+offset_w+stride1_w+stride2_w, nn->conv2d->conv_weights->val+offset_w+stride1_w+stride2_w, (kh*kw)*sizeof(float));
                transpose(nn->conv2d->flipped_weights->val+offset_w+stride1_w+stride2_w, kh, kw);
                reverse_rows(nn->conv2d->flipped_weights->val+offset_w+stride1_w+stride2_w, kh, kw);
                transpose(nn->conv2d->flipped_weights->val+offset_w+stride1_w+stride2_w, kh, kw);
                reverse_rows(nn->conv2d->flipped_weights->val+offset_w+stride1_w+stride2_w, kh, kw);
                stride2_w = stride2_w + (kw * kw);
            }
            stride1_w = stride1_w + (q * kh * kw);
        }
        offset_w = offset_w + (p * q * kh * kw);
        offset_f = offset_f + (kw * kw);
    }
}

//
// This routine flips horizontally and vertically the deltas (errors) at a given convolutional layer.
// It operates on the latest updated values in deltas_buffer and stores the result in propag_buffer
//
void flipDeltas(void * _Nonnull neural, unsigned int q, unsigned fh, unsigned int fw) {
    
    extern tensor *propag_buffer;
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    int offset = 0;
    for (int k=0; k<q; k++) {
        int indx = 0;
        int stride = offset + (fh * fw);
        for (int i=stride-1; i>=offset; i--) {
            propag_buffer->val[offset+indx] = nn->conv2d->deltas_buffer->val[i];
            indx++;
        }
        offset = offset + (fh * fw);
    }
}

//
// This routine updates the kernel matrices using the flipped kernels (weights)
//
void kernelMatUpdate(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int idx = 0;
    int offset_km = 0;
    int offset_w = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
            
            unsigned int p = nn->conv2d->conv_weights->shape[idx][0][0];
            unsigned int q = nn->conv2d->conv_weights->shape[idx][1][0];
            unsigned int kh = nn->conv2d->conv_weights->shape[idx][2][0];
            unsigned int kw = nn->conv2d->conv_weights->shape[idx][3][0];
            
            unsigned int kmh = nn->conv2d->kernel_matrices->shape[idx][0][0];
            unsigned int kmw = nn->conv2d->kernel_matrices->shape[idx][1][0];
            
            int stride1 = 0;
            int m = 0;
            for (int k=0; k<p*(kh*kw); k++) {
                int stride2 = 0;
                for (int ll=0; ll<q; ll++) {
                    nn->conv2d->kernel_matrices->val[offset_km+(k*q+ll)] = nn->conv2d->flipped_weights->val[offset_w+(stride1+(stride2+m))];
                    stride2 = stride2 + (kh * kw);
                }
                m++;
                if (m == (kh * kw)) {
                    stride1 = stride1 + (q * kh * kw);
                     m = 0;
                }
            }
            offset_w = offset_w + (p * q * kh * kw);
            offset_km = offset_km + (kmh * kmw);
            idx++;
        }
    }
}



void trainLoop(void * _Nonnull  neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    float **miniBatch = floatmatrix(0, nn->dense->parameters->miniBatchSize-1, 0, nn->data->training->n-1);
    float out_test[nn->data->test->m];
    
    for (int k=1; k<=nn->dense->parameters->epochs; k++) {
        shuffle(nn->data->training->set, nn->data->training->m, nn->data->training->n);
        
        for (int l=1; l<=nn->dense->train->batch_range((void *)neural,  nn->dense->parameters->miniBatchSize); l++) {
            nn->dense->train->next_batch((void *)nn, miniBatch, nn->dense->parameters->miniBatchSize);
            
            if (nn->dense->train->gradient_descent != NULL) {
                nn->dense->train->gradient_descent->minimize((void *)nn, miniBatch, nn->dense->parameters->miniBatchSize);
            } else if (nn->dense->train->momentum != NULL) {
                nn->dense->train->momentum->minimize((void *)nn, miniBatch, nn->dense->parameters->miniBatchSize);
            } else if (nn->dense->train->ada_grad != NULL) {
                nn->dense->train->ada_grad->minimize((void *)nn, miniBatch, nn->dense->parameters->miniBatchSize);
            } else if (nn->dense->train->rms_prop != NULL) {
                nn->dense->train->rms_prop->minimize((void *)nn, miniBatch, nn->dense->parameters->miniBatchSize);
            }  else if (nn->dense->train->adam != NULL) {
                nn->dense->train->adam->minimize((void *)nn, miniBatch, nn->dense->parameters->miniBatchSize);
            }
        }
        
        fprintf(stdout, "%s: Epoch {%d/%d}: testing network with {%u} inputs:\n", DEFAULT_CONSOLE_WRITER, k, nn->dense->parameters->epochs, nn->data->test->m);
        nn->eval_prediction((void *)nn, "test", out_test, false);
        float acc_test = nn->math_ops(out_test, nn->data->test->m, "reduce sum");
        fprintf(stdout, "{%d/%d}: Test accuracy: %d/%d.\n", k, nn->dense->parameters->epochs, (int)acc_test, nn->data->test->m);
        fprintf(stdout, "\n");
    }
    
    free_fmatrix(miniBatch, 0, nn->dense->parameters->miniBatchSize-1, 0, nn->data->training->n-1);
}
