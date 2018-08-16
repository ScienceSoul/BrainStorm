//
//  Conv2DNetOps.c
//  BrainStorm
//
//  Created by Hakime Seddik on 13/08/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include "cblas.h"
#endif

#include "NeuralNetwork.h"
#include "Conv2DNetOps.h"

static unsigned int offset_w;
static unsigned int offset_a;
static unsigned int offset_a_compute;
static unsigned int offset_b;

static unsigned int offset_a_connected;
static unsigned int offset_a_connected_compute;
static unsigned int offset_w_connected;
static unsigned int offset_b_connected;
static unsigned int offset_z_connected;

void convolution_ops(void * _Nonnull  neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    unsigned int p = nn->conv2d->parameters->topology[layer-1][1];
    unsigned int q = nn->conv2d->parameters->topology[layer][1];
    
    unsigned int fh = nn->conv2d->parameters->topology[layer][2];
    unsigned int fw = nn->conv2d->parameters->topology[layer][3];
    unsigned int kh = nn->conv2d->parameters->topology[layer][4];
    unsigned int kw = nn->conv2d->parameters->topology[layer][5];
    unsigned int sh = nn->conv2d->parameters->topology[layer][6];
    unsigned int sw = nn->conv2d->parameters->topology[layer][7];
    
    static unsigned int local_idx = 0;
    if (advance > 0) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_weights->rank; i++) {
            step = step * nn->conv2d->conv_weights->shape[*advance-1][i][0];
        }
        offset_w = offset_w + step;
        
        step = 1;
        for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
            step = step * nn->conv2d->conv_activations->shape[*advance-1][i][0];
        }
        offset_a = offset_a + step;
        
        step = 1;
        for (int i=0; i<nn->conv2d->conv_biases->rank; i++) {
            step = step * nn->conv2d->conv_biases->shape[*advance-1][i][0];
        }
        offset_b = offset_b + step;
        
    } else {
        offset_w = 0;
        offset_a = 0;
        offset_a_compute = 0;
        offset_b = 0;
    }
    
    // Offset to activations at current convolution layer
    int step = 1;
    for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
        step = step * nn->conv2d->conv_activations->shape[*advance][i][0];
    }
    offset_a_compute = offset_a_compute + step;
    
    int row_order_a = nn->conv2d->conv_activations->shape[*advance][2][0];
    int col_order_a = nn->conv2d->conv_activations->shape[*advance][1][0];
    int row_order_a_compute = nn->conv2d->parameters->topology[layer][3];
    
    int row_order_w = nn->conv2d->conv_weights->shape[local_idx][3][0];
    int col_order_w = nn->conv2d->conv_weights->shape[local_idx][2][0];
    
    int stride2_w = 0;
    int stride_a_compute = 0;
    for (int k=0; k<q; k++) {// Loop over all feature maps at current layer
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                float sum = 0.0f;
                int stride_a = 0;
                int stride1_w = 0;
                for (int l=0; l<p; l++) { // Loop over all feature maps in previous layer
                    for (int u=0; u<kh; u++) {
                        for (int v=0; v<kw; v++) {
                            sum = sum +
                            (nn->conv2d->conv_activations->val[offset_a+(stride_a+(((i*sh+u)*row_order_a)+(j*sw+v)))] * nn->conv2d->conv_weights->val[offset_w+(stride1_w+(stride2_w+((u*row_order_w)+v)))] + nn->conv2d->conv_biases->val[offset_b+k]);
                        }
                    }
                    stride1_w = stride1_w + (nn->conv2d->conv_weights->shape[l][1][0] * col_order_w * row_order_w);
                    stride_a = stride_a + (row_order_a * col_order_a);
                }
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*row_order_a_compute)+j))] = nn->conv2d->activationFunctions[local_idx](sum, NULL, NULL);
            }
        }
        stride2_w = stride2_w + (col_order_w * row_order_w);
        stride_a_compute = stride_a_compute + (fh * fw);
    }
    local_idx++;
}

void max_pool(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    unsigned int q = nn->conv2d->parameters->topology[layer][1];
    
    unsigned int fh = nn->conv2d->parameters->topology[layer][2];
    unsigned int fw = nn->conv2d->parameters->topology[layer][3];
    unsigned int kh = nn->conv2d->parameters->topology[layer][4];
    unsigned int kw = nn->conv2d->parameters->topology[layer][5];
    unsigned int sh = nn->conv2d->parameters->topology[layer][6];
    unsigned int sw = nn->conv2d->parameters->topology[layer][7];
    
    if (advance > 0) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
            step = step * nn->conv2d->conv_activations->shape[*advance-1][i][0];
        }
        offset_a = offset_a + step;
    } else {
        offset_w = 0;
        offset_a = 0;
        offset_a_compute = 0;
        offset_b = 0;
    }
    
    // Offset to activations at current pooling layer
    int step = 1;
    for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
        step = step * nn->conv2d->conv_activations->shape[*advance][i][0];
    }
    offset_a_compute = offset_a_compute + step;
    
    int row_order_a = nn->conv2d->conv_activations->shape[*advance][2][0];
    int col_order_a = nn->conv2d->conv_activations->shape[*advance][1][0];
    int row_order_a_compute = nn->conv2d->parameters->topology[layer][3];
    
    int stride_a_compute = 0;
    int stride_a = 0;
    float values[kh*kw];
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                int idx = 0;
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        values[idx] = nn->conv2d->conv_activations->val[offset_a+(stride_a+(((2*i*sh+u)*row_order_a)+(2*j*sw+v)))];
                        idx++;
                    }
                }
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*row_order_a_compute)+j))] = max_array(values, kh*kw);
            }
        }
        stride_a = stride_a + (col_order_a * row_order_a);
        stride_a_compute = stride_a_compute + (kh * kw);
    }
}

void l2_pool(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    unsigned int q = nn->conv2d->parameters->topology[layer][1];
    
    unsigned int fh = nn->conv2d->parameters->topology[layer][2];
    unsigned int fw = nn->conv2d->parameters->topology[layer][3];
    unsigned int kh = nn->conv2d->parameters->topology[layer][4];
    unsigned int kw = nn->conv2d->parameters->topology[layer][5];
    unsigned int sh = nn->conv2d->parameters->topology[layer][6];
    unsigned int sw = nn->conv2d->parameters->topology[layer][7];
    
    if (advance > 0) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
            step = step * nn->conv2d->conv_activations->shape[*advance-1][i][0];
        }
        offset_a = offset_a + step;
    } else {
        offset_w = 0;
        offset_a = 0;
        offset_a_compute = 0;
        offset_b = 0;
    }
    
    int step = 1;
    for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
        step = step * nn->conv2d->conv_activations->shape[*advance][i][0];
    }
    offset_a_compute = offset_a_compute + step;
    
    int row_order_a = nn->conv2d->conv_activations->shape[*advance][2][0];
    int col_order_a = nn->conv2d->conv_activations->shape[*advance][1][0];
    int row_order_a_compute = nn->conv2d->parameters->topology[layer][3];
    
    int stride_a_compute = 0;
    int stride_a = 0;
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                float sum = 0.0f;
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        sum = sum + (nn->conv2d->conv_activations->val[offset_a+(stride_a+(((2*i*sh+u)*row_order_a)+(2*j*sw+v)))] * nn->conv2d->conv_activations->val[offset_a+(stride_a+(((2*i*sh+u)*row_order_a)+(2*j*sw+v)))]);
                    }
                }
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*row_order_a_compute)+j))] = sqrtf(sum);
            }
        }
        stride_a = stride_a + (col_order_a * row_order_a);
        stride_a_compute = stride_a_compute + (kh * kw);
    }
}

void average_pool(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    unsigned int q = nn->conv2d->parameters->topology[layer][1];
    
    unsigned int fh = nn->conv2d->parameters->topology[layer][2];
    unsigned int fw = nn->conv2d->parameters->topology[layer][3];
    unsigned int kh = nn->conv2d->parameters->topology[layer][4];
    unsigned int kw = nn->conv2d->parameters->topology[layer][5];
    unsigned int sh = nn->conv2d->parameters->topology[layer][6];
    unsigned int sw = nn->conv2d->parameters->topology[layer][7];
    
    if (advance > 0) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
            step = step * nn->conv2d->conv_activations->shape[*advance-1][i][0];
        }
        offset_a = offset_a + step;
    } else {
        offset_w = 0;
        offset_a = 0;
        offset_a_compute = 0;
        offset_b = 0;
    }
    
    int step = 1;
    for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
        step = step * nn->conv2d->conv_activations->shape[*advance][i][0];
    }
    offset_a_compute = offset_a_compute + step;
    
    int row_order_a = nn->conv2d->conv_activations->shape[*advance][2][0];
    int col_order_a = nn->conv2d->conv_activations->shape[*advance][1][0];
    int row_order_a_compute = nn->conv2d->parameters->topology[layer][3];
    
    int stride_a_compute = 0;
    int stride_a = 0;
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                float sum = 0.0f;
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        sum = sum +
                        nn->conv2d->conv_activations->val[offset_a+(stride_a+(((2*i*sh+u)*row_order_a)+(2*j*sw+v)))];
                    }
                }
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*row_order_a_compute)+j))] = sum / (kh*kw);
            }
        }
        stride_a = stride_a + (col_order_a * row_order_a);
        stride_a_compute = stride_a_compute + (kh * kw);
    }
}

void full_connected_ops(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    static bool advance_connected_a;
    static bool advance_connected_compute;
    static int local_idx;
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    tensor *ptr_activations = NULL;
    unsigned int *ptr_offset = NULL;
    
    if (advance > 0) {
        if (nn->conv2d->parameters->topology[layer-1][0] == POOLING ||
            nn->conv2d->parameters->topology[layer-1][0] == CONVOLUTION) {
            int step = 1;
            for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
                step = step * nn->conv2d->conv_activations->shape[*advance-1][i][0];
            }
            offset_a = offset_a + step;
            ptr_activations = nn->conv2d->conv_activations;
            ptr_offset = &offset_a;
            
            offset_a_connected = 0;
            advance_connected_a = false;
            advance_connected_compute = false;
            local_idx = 0;
        } else {
            if (!advance_connected_a) {
                advance_connected_a = true;
            } else {
                offset_a_connected = offset_a_connected + nn->conv2d->dense_activations->shape[local_idx-1][0][0];
            }
            ptr_activations = nn->conv2d->dense_activations;
            ptr_offset = &offset_a_connected;
            local_idx++;
        }
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "topology error in convolutional network. Fully connected layer operation coming too early in the network operations stack.");
    }
    
    if (!advance_connected_compute) {
        offset_a_connected_compute = 0;
        offset_w_connected = 0;
        offset_b_connected = 0;
        offset_z_connected = 0;
        advance_connected_compute = true;
    } else {
        offset_a_connected_compute = offset_a_connected_compute + nn->conv2d->dense_activations->shape[local_idx-1][0][0];
        offset_w_connected = offset_w_connected +
             (nn->conv2d->dense_weights->shape[local_idx-1][0][0]*nn->conv2d->dense_weights->shape[local_idx-1][1][0]);
        offset_b_connected = offset_b_connected + nn->conv2d->dense_biases->shape[local_idx-1][0][0];
        
        offset_z_connected = offset_z_connected + nn->conv2d->dense_affineTransformations->shape[local_idx-1][0][0];
    }
    
    float buffer[nn->conv2d->dense_activations->shape[local_idx][0][0]];
    memset(buffer, 0.0f, sizeof(buffer));
    
    unsigned int m = nn->conv2d->dense_weights->shape[local_idx][0][0];
    unsigned int n = nn->conv2d->dense_weights->shape[local_idx][1][0];
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n, 1.0, nn->conv2d->dense_weights->val+offset_w_connected, (int)n, ptr_activations->val+*ptr_offset, 1, 0.0, buffer, 1);
#ifdef __APPLE__
    vDSP_vadd(buffer, 1, nn->conv2d->dense_biases->val+offset_b_connected, 1, nn->conv2d->dense_affineTransformations->val+offset_z_connected, 1, nn->conv2d->dense_biases->shape[local_idx][0][0]);
#else
    for (int i=0; i<nn->conv2d->dense_biases->shape[layer_index][0][0]; i++) {
        nn->conv2d->dense_affineTransformations->val[offset_z_connected+i] = buffer[i] + nn->conv2d->dense_biases->val[offset_b_connected+i];
    }
#endif
    
    float *vec = NULL;
    unsigned int *vec_length = NULL;
    // To get the activation function associated with a fully connected layer, we assume that
    // the fully connected layers always come after all convolution layers
    // (no interleaving between convolution layers) which should be the case for a correctly
    // constructed convolutional network. Otherwise wrong behavior and results!!
    if (nn->activationFunctionsRef[local_idx+nn->conv2d->num_conv2d_layers] == SOFTMAX) {
        vec = nn->conv2d->dense_affineTransformations->val+offset_z_connected;
        vec_length = &(nn->conv2d->dense_affineTransformations->shape[local_idx][0][0]);
    }
    for (int i=0; i<nn->conv2d->dense_activations->shape[local_idx][0][0]; i++) {
        nn->conv2d->dense_activations->val[offset_a_connected_compute+i] =
             nn->conv2d->activationFunctions[local_idx+nn->conv2d->num_conv2d_layers](nn->conv2d->dense_affineTransformations->val[offset_z_connected+i], vec, vec_length);
    }
    
    nanToNum(nn->conv2d->dense_activations->val+offset_a_connected_compute, nn->conv2d->dense_activations->shape[local_idx][0][0]);
}

void inference_in_conv2d_net(void * _Nonnull neural) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    unsigned int ops_idx = 0;
    unsigned int advance = 0;
    unsigned int infer_layers = nn->conv2d->num_conv2d_layers+nn->conv2d->num_pooling_layers+nn->conv2d->num_dense_layers;
    for (int l=1; l<=infer_layers; l++) {
        nn->conv2d->layersOps[ops_idx]((void *)nn, l, &advance);
        ops_idx++;
        advance++;
    }
}

void backpropag_in_conv2d_net(void * _Nonnull neural,
                              void (* _Nullable ptr_inference_func)(void * _Nonnull self)) {
    
}

void batch_accumulation_in_conv2d_net(void * _Nonnull neural) {
    
}
