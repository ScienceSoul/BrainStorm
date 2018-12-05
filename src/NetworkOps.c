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

#ifdef __linux__
    #include <bsd/stdlib.h>
#endif

#include "NeuralNetwork.h"
#include "NetworkOps.h"
#include "DenseNetOps.h"
#include "Conv2DNetOps.h"
#include "Memory.h"

static int remainder_offsets[3];
typedef void (*eval_net_type)(void * _Nonnull neural,  tensor * _Nonnull inputs, tensor * _Nonnull labels, float * _Nonnull out);

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

void nextBatch(void * _Nonnull neural, tensor * _Nonnull features, tensor * _Nonnull labels, unsigned int batchSize, int * _Nullable remainder, bool do_remainder) {
    
    static bool firstTime = true;
    static int delta1 = 0;
    static int delta2 = 0;
    static int count = 1;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    tensor *t1 = (tensor *)nn->data->training->set;
    tensor *t2 = (tensor *)nn->data->training->labels;
    
    static int dim1 = 0;
    static int dim2 = 0;
    static int num_inputs = 0;
    if (firstTime) {
        dim2 = 1;
        if (nn->is_dense_network) {
            dim1 = nn->dense->parameters->topology[0];
            if (nn->dense->parameters->numberOfClassifications > 0) {
                dim2 = nn->dense->parameters->numberOfClassifications;
            }
            
            num_inputs = t1->shape[0][0][0] / dim1;
            
        } else if (nn->is_conv2d_network) {
            dim1 = t1->shape[0][1][0] * t1->shape[0][2][0] * t1->shape[0][3][0];
            if (nn->conv2d->parameters->numberOfClassifications > 0) {
                dim2 = nn->conv2d->parameters->numberOfClassifications;
            }
            
            num_inputs = t1->shape[0][0][0];
        }
        
        memset(remainder_offsets, 0, sizeof(remainder_offsets));
        firstTime = false;
    }
    
    if (do_remainder) {
        
        int copy_1 = num_inputs - remainder_offsets[0];
        int copy_2 = batchSize - copy_1;
        
        fprintf(stdout, "%s: remaining %d and will sample %d examples for next mini-batch.\n", DEFAULT_CONSOLE_WRITER, copy_1, copy_2);
        
        // Get the remaining inputs
        memcpy(features->val, t1->val+remainder_offsets[1], (copy_1*dim1)*sizeof(float));
        memcpy(labels->val, t2->val+remainder_offsets[2], (copy_1*dim2)*sizeof(float));
        
        // For the rest of the mini batch, sample randomly from the training set
        for (int i=0; i<copy_2; i++) {
            int idx = arc4random_uniform(num_inputs);
            memcpy(features->val+((copy_1*dim1)+(i*dim1)), t1->val+idx, dim1*sizeof(float));
            memcpy(labels->val+((copy_1*dim2)+(i*dim2)), t2->val+idx, dim2*sizeof(float));
        }
        
        memset(remainder_offsets, 0, sizeof(remainder_offsets));
        return;
        
    } else {
        memcpy(features->val, t1->val+delta1, (batchSize*dim1)*sizeof(float));
        memcpy(labels->val, t2->val+delta2, (batchSize*dim2)*sizeof(float));
    }
    
    if (count == (int)ceil(num_inputs/batchSize)) {
        
        if (remainder != NULL) {
            if (*remainder != 0) {
                remainder_offsets[1] = delta1 + (batchSize * dim1);
                remainder_offsets[2] = delta2 + (batchSize * dim2);
            }
        }
        
        delta1 = 0;
        delta2 = 0;
        count = 1;
    } else {
        delta1 = delta1 + (batchSize * dim1);
        delta2 = delta2 + (batchSize * dim2);
        count++;
    }
    if (!do_remainder && remainder != NULL) remainder_offsets[0] = remainder_offsets[0] + batchSize;
}


int batchRange(void * _Nonnull neural, unsigned int batchSize, int * _Nullable remainder) {
    
    static bool firstTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    tensor *t1 = (tensor *)nn->data->training->set;
    
    static int num_inputs = 0;
    if (firstTime) {
        
        if (nn->is_dense_network) {
            num_inputs = t1->shape[0][0][0] / nn->dense->parameters->topology[0];
        } else if (nn->is_conv2d_network) {
            num_inputs = t1->shape[0][0][0];
        }
        
        firstTime = false;
    }
    
    if (remainder != NULL) *remainder = num_inputs % batchSize;
    return (int)ceil((int)num_inputs/batchSize);
}

void progression(void * _Nonnull neural, progress_dict progress_dict) {
    
    static bool firstTime = true;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    static int num_inputs = 0;
    if (firstTime) {
        
        tensor *t = (tensor *)nn->data->training->set;
        if (nn->is_dense_network) {
            num_inputs = t->shape[0][0][0] / nn->dense->parameters->topology[0];
        } else if (nn->is_conv2d_network) {
            num_inputs = t->shape[0][0][0];
        }
        
        firstTime = false;
    }
    
    int train_size = (int)num_inputs/progress_dict.batch_size;
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
    
    if (count == (int)ceil((int)num_inputs/progress_dict.batch_size)) {
        i = 0;
        nextPrint = step;
        count = 1;
    } else count++;
}

float mathOps(float * _Nonnull vector, unsigned int n, char * _Nonnull op) {
    
    float result = 0.0f;
    
    if (strcmp(op, "reduce mean") == 0) {
#ifdef __APPLE__
        vDSP_meanv(vector, 1, &result, n);
#else
        result = meanv(vector, n);
#endif
        
    } else if (strcmp(op, "reduce sum") == 0) {
#ifdef __APPLE__
        vDSP_sve(vector, 1, &result, n);
#else
        result = sve(vector, n);
#endif

    } else if (strcmp(op, "reduce max") == 0) {
#ifdef __APPLE__
        vDSP_maxv(vector, 1, &result, n);
#else
        result = maxv(vector, n);
#endif
        
    } else if (strcmp(op, "reduce min") == 0) {
#ifdef _APPLE__
        vDSP_minv(vector, 1, &result, n);
#else
        result = minv(vector, n);
#endif
    } else fatal(DEFAULT_CONSOLE_WRITER, "unrecognized math operation.");
    
    return result;
}

void eval_dense_net(void * _Nonnull neural,  tensor * _Nonnull inputs, tensor * _Nonnull labels, float * _Nonnull out) {
 
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Offset to activations at output layer
    unsigned int offset = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        offset = offset + nn->dense->activations->shape[l][0][0];
    }
    
    for (int k=0; k<inputs->shape[0][0][0]/nn->dense->parameters->topology[0]; k++) {
        
        int stride = k * nn->dense->parameters->topology[0];
        memcpy(nn->dense->activations->val, inputs->val+stride, (nn->dense->parameters->topology[0])*sizeof(float));
        
        inference_in_dense_net(nn);
        
        out[k] = (float)argmax(nn->dense->activations->val+offset, nn->dense->activations->shape[nn->network_num_layers-1][0][0]) == labels->val[k];
    }
}

void eval_conv2d_net(void * _Nonnull neural,  tensor * _Nonnull inputs, tensor * _Nonnull labels, float * _Nonnull out) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Offset to activations at the output layer
    unsigned int offset = 0;
    for (int l=0; l<nn->conv2d->num_dense_layers-1; l++) {
        offset = offset + nn->conv2d->dense_activations->shape[l][0][0];
    }
    
    for (int k=0; k<inputs->shape[0][0][0]; k++) {
    
        //TODO: channels > 1?
        int fh = inputs->shape[0][1][0];
        int fw = inputs->shape[0][2][0];
        int channels = inputs->shape[0][3][0];
        int stride1 = k * (fh * fw * channels);
        memcpy(nn->conv2d->conv_activations->val, inputs->val+stride1, (fh*fw*channels)*sizeof(float));
        
        inference_in_conv2d_net(nn);
        
        out[k] = (float)argmax(nn->conv2d->dense_activations->val+offset, nn->conv2d->dense_activations->shape[nn->conv2d->num_dense_layers-1][0][0]) == labels->val[k];
    }
}

static void eval(void * _Nonnull self, tensor * _Nonnull inputs, tensor * _Nonnull labels, float * _Nonnull out) {
    
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
    
    eval_net(self, inputs, labels, out);
}

void evalPrediction(void * _Nonnull self, char * _Nonnull dataSet, float * _Nonnull out, bool metal) {
    
    static bool test_check = false;
    static bool validation_check = false;
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    tensor *t1 = NULL;
    tensor *t2 = NULL;
    if (strcmp(dataSet, "validation") == 0) {
        if (!validation_check) {
            if (nn->data->validation->set == NULL) fatal(DEFAULT_CONSOLE_WRITER, "trying to evaluate prediction on validation data but the data do not exist.");
            validation_check = true;
        }
        
        t1 = (tensor *)nn->data->validation->set;
        t2 = (tensor *)nn->data->validation->labels;
    } else if (strcmp(dataSet, "test") == 0) {
        if (!test_check) {
            if (nn->data->test->set == NULL) fatal(DEFAULT_CONSOLE_WRITER, "trying to evaluate prediction on test data but the data do not exist.");
            test_check = true;
        }
        
        t1 = (tensor *)nn->data->test->set;
        t2 = (tensor *)nn->data->test->labels;
    } else fatal(DEFAULT_CONSOLE_WRITER, "unrecognized data set in prediction evaluation.");
    
#ifdef __APPLE__
    if (metal) {
        fatal(DEFAULT_CONSOLE_WRITER, "Offload evaluation to GPU broken.");
#ifdef GPU_WORKING
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
#endif
    } else {
        eval(self, t1, t2, out);
    }
#else
    eval(self, t1, t2, out);
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
    
    tensor *t1 = NULL;
    tensor *t2 = NULL;
    if (strcmp(dataSet, "training") == 0) {
        
        t1 = (tensor *)nn->data->training->set;
        t2 = (tensor *)nn->data->training->labels;
    } else if (strcmp(dataSet, "validation") == 0) {
        if (!validation_check) {
            if (nn->data->validation->set == NULL) fatal(DEFAULT_CONSOLE_WRITER, "trying to evaluate cost on validation data but the data do not exist.");
            validation_check = true;
        }
        
        t1 = (tensor *)nn->data->validation->set;
        t2 = (tensor *)nn->data->validation->labels;
    } else if (strcmp(dataSet, "test") == 0) {
        if (!test_check) {
            if (nn->data->test->set == NULL) fatal(DEFAULT_CONSOLE_WRITER, "trying to evaluate cost on test data but the data do not exist.");
            test_check = true;
        }
        
        t1 = (tensor *)nn->data->test->set;
        t2 = (tensor *)nn->data->test->labels;
    } else fatal(DEFAULT_CONSOLE_WRITER, "unrecognized data set in cost evaluation.");
    
    //float norm, sum;
    
    float cost = 0.0f;
//    for (int i=0; i<data_size; i++) {
//
//        for (int j=0; j<nn->num_channels; j++) {
//            nn->dense->activations->val[j] = data[i][j];
//        }
//
//        inference(self);
//
//        // Stride to activations at last layer
//        unsigned stride1 = 0;
//        for (int l=0; l<nn->network_num_layers-1; l++) {
//            stride1 = stride1 + nn->dense->activations->shape[l][0][0];
//        }
//
//        float y[nn->dense->activations->shape[nn->network_num_layers-1][0][0]];
//        memset(y, 0.0f, sizeof(y));
//        if (binarization == true) {
//            for (int j=0; j<nn->dense->activations->shape[nn->network_num_layers-1][0][0]; j++) {
//                if (data[i][nn->num_channels] == nn->dense->parameters->classifications[j]) {
//                    y[j] = 1.0f;
//                }
//            }
//        } else {
//            int idx = (int)nn->num_channels;
//            for (int j=0; j<nn->dense->activations->shape[nn->network_num_layers-1][0][0]; j++) {
//                y[j] = data[i][idx];
//                idx++;
//            }
//        }
//        cost = cost + crossEntropyCost(nn->dense->activations->val+stride1, y, nn->dense->activations->shape[nn->network_num_layers-1][0][0]) / data_size;
//
//        sum = 0.0f;
//        unsigned int stride = 0;
//        for (int l=0; l<nn->network_num_layers-1; l++) {
//            unsigned int m = nn->dense->weights->shape[l][0][0];
//            unsigned int n = nn->dense->weights->shape[l][1][0];
//            norm = frobeniusNorm(nn->dense->weights->val+stride, (m * n));
//            sum = sum + (norm*norm);
//            stride = stride + (m * n);
//        }
//        cost = cost + 0.5f*(nn->dense->parameters->lambda/(float)data_size)*sum;
//    }
    
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
    
    int offset_km = 0;
    int offset_w = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
            
            unsigned int p = nn->conv2d->conv_weights->shape[l][0][0];
            unsigned int q = nn->conv2d->conv_weights->shape[l][1][0];
            unsigned int kh = nn->conv2d->conv_weights->shape[l][2][0];
            unsigned int kw = nn->conv2d->conv_weights->shape[l][3][0];
            
            unsigned int kmh = nn->conv2d->kernel_matrices->shape[l][0][0];
            unsigned int kmw = nn->conv2d->kernel_matrices->shape[l][1][0];
            
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
    }
}

//
// This routine updates the convolution matrices using the flipped kernels (weights)
//
//void convMatUpdate(void * _Nonnull neural) {
//    
//    BrainStormNet *nn = (BrainStormNet *)neural;
//    
//    // Copy the values to the convolution matrices
//    unsigned int idx = 0;
//    int offset_cm = 0;
//    int offset_w = 0;
//    for (int l=0; l<nn->network_num_layers; l++) {
//        if (nn->conv2d->parameters->topology[l][0] == CONVOLUTION) {
//            
//            unsigned int p = nn->conv2d->conv_weights->shape[idx][0][0];
//            unsigned int q = nn->conv2d->conv_weights->shape[idx][1][0];
//            unsigned int kh = nn->conv2d->conv_weights->shape[idx][2][0];
//            unsigned int kw = nn->conv2d->conv_weights->shape[idx][3][0];
//            unsigned int sh = nn->conv2d->parameters->topology[l][6];
//            unsigned int sw = nn->conv2d->parameters->topology[l][7];
//            
//            unsigned int mh = nn->conv2d->conv_matrices->shape[idx][2][0];
//            unsigned int mw = nn->conv2d->conv_matrices->shape[idx][3][0];
//            
//            unsigned int n = nn->conv2d->parameters->topology[l-1][3];
//            unsigned int n_c = nn->conv2d->parameters->topology[l][3];
//            unsigned int after_zeros_step = n - kw;
//            
//            int stride1_cm = 0;
//            int stride1_w = 0;
//            for (int k=0; k<p; k++) {
//                int stride2_cm = 0;
//                int stride2_w = 0;
//                for (int ll=0; ll<q; ll++) {
//                    int step_before_next_row = 0;
//                    int left_offset_sh = 0;
//                    int left_offset_sw = 0;
//                    for (int i = 0; i<mh; i++) {
//                        int pos = 0;
//                        for (int ii=0; ii<kh; ii++) {
//                            for (int jj=0; jj<kw; jj++) {
//                                nn->conv2d->conv_matrices->val[offset_cm+(stride1_cm+(stride2_cm+(((i*mw)+left_offset_sw+left_offset_sh*sh)+after_zeros_step*ii+pos)))] = nn->conv2d->flipped_weights->val[offset_w+(stride1_w+(stride2_w+((ii*kw)+jj)))];
//                                pos++;
//                            }
//                        }
//                        step_before_next_row++;
//                        left_offset_sh++;
//                        if (step_before_next_row == n_c) {
//                            left_offset_sh = 0;
//                            step_before_next_row = 0;
//                            left_offset_sw = left_offset_sw + (sw * n);
//                        }
//                    }
//                    stride2_cm = stride2_cm + (mh * mw);
//                    stride2_w = stride2_w + (kh * kw);
//                }
//                stride1_cm = stride1_cm + (q * mh * mw);
//                stride1_w = stride1_w + (q * kh * kw);
//            }
//            
//            idx++;
//            offset_cm = offset_cm + (p * q * mh * mw);
//            offset_w = offset_w + (p * q * kh * kw);
//        }
//    }
//}


void trainLoop(void * _Nonnull  neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    tensor_dict *dict = init_tensor_dict();
    dict->rank = 1;
    dict->shape[0][0][0] = nn->dense->parameters->miniBatchSize * nn->dense->parameters->topology[0];
    tensor *features = (tensor *)nn->tensor(NULL, *dict);
    
    dict->rank = 1;
    dict->shape[0][0][0] = nn->dense->parameters->miniBatchSize * nn->dense->parameters->numberOfClassifications;
    tensor *labels = (tensor *)nn->tensor(NULL, *dict);
    
    for (int k=1; k<=nn->dense->parameters->epochs; k++) {
        int num_inputs = nn->dense->parameters->topology[0];
        shuffle(nn->data->training->set, nn->data->training->labels, nn->dense->parameters->numberOfClassifications, &num_inputs);
        
        for (int l=1; l<=nn->dense->train->batch_range((void *)neural,  nn->dense->parameters->miniBatchSize, NULL); l++) {
            nn->dense->train->next_batch((void *)neural, features, labels, nn->dense->parameters->miniBatchSize, NULL, false);
            
            if (nn->dense->train->gradient_descent != NULL) {
                nn->dense->train->gradient_descent->minimize((void *)nn, features, labels, nn->dense->parameters->miniBatchSize);
            } else if (nn->dense->train->momentum != NULL) {
                nn->dense->train->momentum->minimize((void *)nn, features, labels, nn->dense->parameters->miniBatchSize);
            } else if (nn->dense->train->ada_grad != NULL) {
                nn->dense->train->ada_grad->minimize((void *)nn, features, labels, nn->dense->parameters->miniBatchSize);
            } else if (nn->dense->train->rms_prop != NULL) {
                nn->dense->train->rms_prop->minimize((void *)nn, features, labels, nn->dense->parameters->miniBatchSize);
            }  else if (nn->dense->train->adam != NULL) {
                nn->dense->train->adam->minimize((void *)nn, features, labels, nn->dense->parameters->miniBatchSize);
            }
        }
        
        tensor *t = (tensor *)nn->data->test->set;
        float out_test[t->shape[0][0][0]/nn->dense->parameters->topology[0]];
        fprintf(stdout, "%s: Epoch {%d/%d}: testing network with {%u} inputs:\n", DEFAULT_CONSOLE_WRITER, k, nn->dense->parameters->epochs, t->shape[0][0][0]/nn->dense->parameters->topology[0]);
        nn->eval_prediction((void *)nn, "test", out_test, false);
        float acc_test = nn->math_ops(out_test, t->shape[0][0][0]/nn->dense->parameters->topology[0], "reduce sum");
        fprintf(stdout, "{%d/%d}: Test accuracy: %d/%d.\n", k, nn->dense->parameters->epochs, (int)acc_test, t->shape[0][0][0]/nn->dense->parameters->topology[0]);
        fprintf(stdout, "\n");
    }
    
    free(features->val);
    free(features);
    
    free(labels->val);
    free(labels);
    
    free(dict);
}
