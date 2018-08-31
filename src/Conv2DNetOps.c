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
static unsigned int offset_dcdw_connected;
static unsigned int offset_dcdb_connected;

static unsigned int activ_idx;

void infer_convolution_op(void * _Nonnull  neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
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

void max_pooling_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
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
                        values[idx] = nn->conv2d->conv_activations->val[offset_a+(stride_a+(((i*sh+u)*row_order_a)+(j*sw+v)))];
                        idx++;
                    }
                }
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*row_order_a_compute)+j))] = max_array(values, kh*kw);
            }
        }
        stride_a = stride_a + (col_order_a * row_order_a);
        stride_a_compute = stride_a_compute + (fh * fw);
    }
}

void l2_pooling_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
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
                        sum = sum + (nn->conv2d->conv_activations->val[offset_a+(stride_a+(((i*sh+u)*row_order_a)+(j*sw+v)))] * nn->conv2d->conv_activations->val[offset_a+(stride_a+(((i*sh+u)*row_order_a)+(j*sw+v)))]);
                    }
                }
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*row_order_a_compute)+j))] = sqrtf(sum);
            }
        }
        stride_a = stride_a + (col_order_a * row_order_a);
        stride_a_compute = stride_a_compute + (fh * fw);
    }
}

void average_pooling_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
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
                        nn->conv2d->conv_activations->val[offset_a+(stride_a+(((i*sh+u)*row_order_a)+(j*sw+v)))];
                    }
                }
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*row_order_a_compute)+j))] = sum / (kh*kw);
            }
        }
        stride_a = stride_a + (col_order_a * row_order_a);
        stride_a_compute = stride_a_compute + (fh * fw);
    }
}

void infer_full_connected_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    static bool advance_connected_a;
    static bool advance_connected_compute;
    static int local_idx;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
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
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int ops_idx = 0;
    unsigned int advance = 0;
    for (int l=1; l<=nn->conv2d->num_infer_ops; l++) {
        nn->conv2d->inferenceOps[ops_idx]((void *)nn, l, &advance);
        ops_idx++;
        advance++;
    }
}

//
// This routine does the backward propagation on the fully connected layers
// similarly to the routine used for a fully connected network.
//
void backpropag_full_connected_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    static unsigned int local_idx;
    
    tensor *ptr_activations = NULL;
    unsigned int *ptr_offset = NULL;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (advance > 0) {
        if (nn->conv2d->parameters->topology[layer-1][0] == FULLY_CONNECTED) {
            offset_a_connected = offset_a_connected - nn->conv2d->dense_activations->shape[local_idx-2][0][0];
            ptr_activations = nn->conv2d->dense_activations;
            ptr_offset = &offset_a_connected;
            
        } else { // The previous layer is a convolution/pooling layer
            // Compute the offset to the activations at the last pooling/convolutional layer
            offset_a = 0;
            for (int l=0; l<nn->conv2d->num_conv2d_layers+nn->conv2d->num_pooling_layers; l++) {
                int step = 1;
                for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
                    step = step * nn->conv2d->conv_activations->shape[l][i][0];
                }
                offset_a = offset_a + step;
            }
            ptr_activations = nn->conv2d->conv_activations;
            ptr_offset = &offset_a;
        }
        
        offset_z_connected = offset_z_connected - nn->conv2d->dense_affineTransformations->shape[local_idx-1][0][0];
        offset_dcdw_connected = offset_dcdw_connected - (nn->conv2d->dense_batchCostWeightDeriv->shape[local_idx-1][0][0]*nn->conv2d->dense_batchCostWeightDeriv->shape[local_idx-1][1][0]);
        offset_dcdb_connected = offset_dcdb_connected - nn->conv2d->dense_batchCostBiasDeriv->shape[local_idx-1][0][0];
        local_idx--;
        
    } else {
        if (nn->conv2d->parameters->topology[layer-1][0] == FULLY_CONNECTED) {
            // Stride to activations, affine transformations and dc_db at last layer
            // of fully connected part
            offset_a_connected = 0;
            offset_z_connected = 0;
            offset_dcdb_connected = 0;
            for (int l=0; l<nn->conv2d->num_dense_layers-1; l++) {
                offset_a_connected = offset_a_connected + nn->conv2d->dense_activations->shape[l][0][0];
                offset_z_connected = offset_z_connected + nn->conv2d->dense_affineTransformations->shape[l][0][0];
                offset_dcdb_connected = offset_dcdb_connected + nn->conv2d->dense_biases->shape[l][0][0];
            }
            // Stride to dc/dw at last layer of fully connected part
            offset_dcdw_connected = 0;
            for (int l=0; l<nn->conv2d->num_dense_layers-1; l++) {
                offset_dcdw_connected = offset_dcdw_connected +
                (nn->conv2d->dense_batchCostWeightDeriv->shape[l][0][0]*nn->conv2d->dense_batchCostWeightDeriv->shape[l][1][0]);
            }
            // Stride to weights at last layer of fully connected part
            offset_w_connected = 0;
            for (int l=0; l<nn->conv2d->num_dense_layers-1; l++) {
                offset_w_connected = offset_w_connected + (nn->conv2d->dense_weights->shape[l][0][0]*nn->conv2d->dense_weights->shape[l][1][0]);
            }
            ptr_activations = nn->conv2d->dense_activations;
        } else { // The previous layer is a convolution/pooling layer
            // Compute the offset to the activations at the last pooling/convolutional layer
            offset_a = 0;
            for (int l=0; l<nn->conv2d->num_conv2d_layers+nn->conv2d->num_pooling_layers; l++) {
                int step = 1;
                for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
                    step = step * nn->conv2d->conv_activations->shape[l][i][0];
                }
                offset_a = offset_a + step;
            }
            ptr_activations = nn->conv2d->conv_activations;
            ptr_offset = &offset_a;
            
            offset_a_connected = 0;
            offset_z_connected = 0;
            offset_dcdb_connected = 0;
            offset_dcdw_connected = 0;
        }
        
        memset(nn->conv2d->propag_delta, 0.0f, nn->conv2d->parameters->max_propag_delta_entries*sizeof(float));
        local_idx = nn->conv2d->num_dense_layers - 1;
        activ_idx = nn->conv2d->num_conv2d_layers + nn->conv2d->num_dense_layers - 1;
        
        // Set here the offset to the last convolutional weights. This is will be used later
        // during backpropagation
        offset_w = 0;
        for (int l=0; l<nn->conv2d->num_conv2d_layers-1; l++) {
            int step = 1;
            for (int i=0; i<nn->conv2d->conv_weights->rank; i++) {
                step = step * nn->conv2d->conv_weights->shape[l][i][0];
            }
            offset_w = offset_w + step;
        }
    }
    
    if (layer == nn->network_num_layers - 1) { // Last network layer
        // Compute delta
        int k = (int)nn->num_channels;
        for (int i=0; i<nn->conv2d->dense_activations->shape[nn->conv2d->num_dense_layers-1][0][0]; i++) {
            nn->conv2d->propag_delta[i] = nn->conv2d->dense_activations->val[offset_a_connected+i] - nn->batch[nn->example_idx][k];
            k++;
        }
        
        if (nn->conv2d->parameters->topology[layer-1][0] == FULLY_CONNECTED) {
            offset_a_connected = offset_a_connected - nn->conv2d->dense_activations->shape[nn->conv2d->num_dense_layers-2][0][0];
            ptr_offset = &offset_a_connected;
        }
        
        int n = nn->conv2d->dense_batchCostWeightDeriv->shape[nn->conv2d->num_dense_layers-1][1][0];
        for (int i=0; i<nn->conv2d->dense_batchCostWeightDeriv->shape[nn->conv2d->num_dense_layers-1][0][0]; i++) {
            for (int j=0; j<nn->conv2d->dense_batchCostWeightDeriv->shape[nn->conv2d->num_dense_layers-1][1][0]; j++) {
                nn->conv2d->dense_batchCostWeightDeriv->val[offset_dcdw_connected+((i*n)+j)] = ptr_activations->val[*ptr_offset+j] * nn->conv2d->propag_delta[i];
            }
        }
        for (int i=0; i<nn->conv2d->dense_batchCostBiasDeriv->shape[nn->conv2d->num_dense_layers-1][0][0]; i++) {
            nn->conv2d->dense_batchCostBiasDeriv->val[offset_dcdb_connected+i] = nn->conv2d->propag_delta[i];
        }
    } else { // Otherwise the layers up the fully connected part
        float buffer[nn->conv2d->parameters->max_number_nodes_in_dense_layer];
        memset(buffer, 0.0f, sizeof(buffer));
        
        float sp[nn->conv2d->dense_affineTransformations->shape[local_idx][0][0]];
        for (int i=0; i<nn->conv2d->dense_affineTransformations->shape[local_idx][0][0]; i++) {
            sp[i] = nn->conv2d->activationDerivatives[activ_idx](nn->conv2d->dense_affineTransformations->val[offset_z_connected+i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->conv2d->dense_weights->shape[local_idx+1][0][0], (int)nn->conv2d->dense_weights->shape[local_idx+1][1][0], 1.0, nn->dense->weights->val+offset_w_connected, (int)nn->conv2d->dense_weights->shape[local_idx+1][1][0], nn->conv2d->propag_delta, 1, 0.0, buffer, 1);
        
        for (int i=0; i<nn->conv2d->dense_affineTransformations->shape[local_idx][0][0]; i++) {
            nn->conv2d->propag_delta[i] = buffer[i] * sp[i];
        }
        int m = nn->conv2d->dense_batchCostWeightDeriv->shape[local_idx][0][0];
        int n = nn->conv2d->dense_costWeightDerivatives->shape[local_idx][1][0];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->conv2d->dense_costWeightDerivatives->val[offset_dcdw_connected+((i*n)+j)] = ptr_activations->val[*ptr_offset+j] * nn->conv2d->propag_delta[i];
            }
        }
        
        for (int i=0; i<nn->conv2d->dense_biases->shape[local_idx][0][0]; i++) {
            nn->conv2d->dense_biases->val[offset_dcdb_connected+i] = nn->conv2d->propag_delta[i];
        }
        
        offset_w_connected = offset_w_connected - (nn->conv2d->dense_weights->shape[local_idx][0][0] * nn->conv2d->dense_weights->shape[local_idx][1][0]);
    }
    activ_idx--;
}

void backpropag_convolution_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
}

//
// This routine is called by the backpropagation pooling op when the pooling layer follows a dense layer
// going the network upward.
// It computes the term transpose(w^{l+1}) x delta^{l+1}, where w^{l+1} and delta^{l+1} are the weights
// and errors of the dense layer which follows the pooling layer downward.
//
static void backpropag_pooling_after_fully_connected(void * _Nonnull neural, unsigned int layer) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    int length = nn->conv2d->parameters->topology[layer][1]*nn->conv2d->parameters->topology[layer][2] *
                 nn->conv2d->parameters->topology[layer][3];
    
    float buffer[length];
    memset(buffer, 0.0f, sizeof(buffer));
    offset_w_connected = 0;
    
    cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->conv2d->dense_weights->shape[0][0][0], (int)nn->conv2d->dense_weights->shape[0][1][0], 1.0, nn->dense->weights->val+offset_w_connected, (int)nn->conv2d->dense_weights->shape[0][1][0], nn->conv2d->propag_delta, 1, 0.0, buffer, 1);
    
    memcpy(nn->conv2d->propag_delta, buffer, length*sizeof(float));
}

//
// This routine is called by the backpropagation pooling op when the pooling layer follows a convolution
// layer going the network upward.
// It computes the term delta^{l+1}*rot{k^{l+1}}, where * is the convolution operation, delta^{l+1} and
// k{l+1} are the error (stored in the propag_upsampling array) and the weights of the convolution layer which
// follows the pooling layer downward.
//
static void backpropag_pooling_after_convolution(void * _Nonnull neural, unsigned int layer) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // The flipping matrix
    float flip_mat[nn->conv2d->parameters->topology[layer+1][4]][nn->conv2d->parameters->topology[layer+1][5]];
    memset(*flip_mat, 0.0f, (nn->conv2d->parameters->topology[layer+1][4]*nn->conv2d->parameters->topology[layer+1][5])*sizeof(float));
    for (int i=0; i<nn->conv2d->parameters->topology[layer+1][4]; i++) {
        flip_mat[i][nn->conv2d->parameters->topology[layer+1][4]-i-1] = 1.0f;
    }
    
    unsigned int p = nn->conv2d->parameters->topology[layer][1];
    unsigned int q = nn->conv2d->parameters->topology[layer+1][1];
    
    unsigned int fh = nn->conv2d->parameters->topology[layer][2];
    unsigned int fw = nn->conv2d->parameters->topology[layer][3];
    unsigned int kh = nn->conv2d->parameters->topology[layer][4];
    unsigned int kw = nn->conv2d->parameters->topology[layer][5];
   
    int stride1_w = 0;
    for (int k=0; k<p; k++) { // Loop accross maps in pooling layer
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                int stride2_w = 0;
                for (int l=0; l<q; l++) { // Loop accross maps in convolution layer
                
                }
            }
            //stride2_w =
        }
        //stride1_w =
    }
    
    
    
    
    
    
}

void backpropag_max_pooling_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    static bool check = false;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (!check) {
        if (advance == 0) fatal(DEFAULT_CONSOLE_WRITER, "topology error in convolutional network. Using a pooling layer as an output layer is not permitted.");
        check = true;
    }
    
    if (nn->conv2d->parameters->topology[layer+1][0] == CONVOLUTION) {
        backpropag_pooling_after_convolution((void *)nn, layer);
    } else {
        backpropag_pooling_after_fully_connected((void *)nn, layer);
    }
    
    // Upsampling
    
    activ_idx--;
}

void backpropag_l2_pooling(void * _Nonnull neural, unsigned int layer, unsigned int *_Nullable advance) {
    
}

void backpropag_average_pooling(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance) {
    
    static bool check = false;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (!check) {
        if (advance == 0) fatal(DEFAULT_CONSOLE_WRITER, "topology error in convolutional network. Using a pooling layer as an output layer is not permitted.");
        check = true;
    }
    
    if (nn->conv2d->parameters->topology[layer+1][0] == CONVOLUTION) {
        backpropag_pooling_after_convolution((void *)nn, layer);
    } else {
        backpropag_pooling_after_fully_connected((void *)nn, layer);
    }
    
    // Upsampling: the error is multiplied by 1/(kh*kw) and assigned to the
    // whole pooling block. kh and kw are the horizontal and vertical dimension
    // of the pooling kernel respectively
    unsigned int q = nn->conv2d->parameters->topology[layer-1][1];
    unsigned int fh = nn->conv2d->parameters->topology[layer][2];
    unsigned int fw = nn->conv2d->parameters->topology[layer][3];
    unsigned int kh = nn->conv2d->parameters->topology[layer][4];
    unsigned int kw = nn->conv2d->parameters->topology[layer][5];
    unsigned int sh = nn->conv2d->parameters->topology[layer][6];
    unsigned int sw = nn->conv2d->parameters->topology[layer][7];
    
    unsigned int row_order_c = nn->conv2d->parameters->topology[layer-1][3];
    unsigned int row_order_p = nn->conv2d->parameters->topology[layer][3];
    
    int stride_c = 0;
    int stride_p = 0;
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        nn->conv2d->propag_upsampling[stride_c+(((i*sh+u)*row_order_c)+(j*sw+v))] =
                                1.0/((float)(kh*kw))*nn->conv2d->propag_delta[stride_p+((i*row_order_p)+j)];
                    }
                }
            }
        }
        stride_c = stride_c + (nn->conv2d->parameters->topology[layer-1][2] * nn->conv2d->parameters->topology[layer-1][3]);
        stride_p = stride_p + (nn->conv2d->parameters->topology[layer][2] * nn->conv2d->parameters->topology[layer][3]);
    }
    
    activ_idx--;
}

void backpropag_in_conv2d_net(void * _Nonnull neural,
                              void (* _Nullable ptr_inference_func)(void * _Nonnull self)) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Activations at the input layer
    // TODO: it would be maybe better to have the batch being
    // defined as a 4D tensor of shape [mini-batch size, height, width, channels]
    // Note that inputs with multi-channels are not supported yet
    for (int i=0; i<nn->num_channels; i++) {
        nn->conv2d->conv_activations->val[i] = nn->batch[nn->example_idx][i];
    }
    
    // Feedforward
    inference_in_conv2d_net((void *)neural);
    
    // Backpropagation
    unsigned int advance = 0;
    for (int l=nn->conv2d->num_backpropag_ops; l>=1; l--) {
        
        advance++;
    }
    
}

void batch_accumulation_in_conv2d_net(void * _Nonnull neural) {
    
}
