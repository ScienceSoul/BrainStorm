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

static unsigned int offset_m;
static unsigned int offset_w;
static unsigned int offset_a;
static unsigned int offset_a_compute;
static unsigned int offset_b;
static unsigned int offset_z;
static unsigned int offset_mask;
static unsigned int mask_idx;

static unsigned int offset_a_connected;
static unsigned int offset_a_connected_compute;
static unsigned int offset_w_connected;
static unsigned int offset_b_connected;
static unsigned int offset_z_connected;
static unsigned int offset_dcdw_connected;
static unsigned int offset_dcdb_connected;

static unsigned int activ_idx;

void infer_convolution_op(void * _Nonnull  neural, unsigned int op, unsigned int * _Nullable advance) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int p = nn->conv2d->parameters->topology[op-1][1];
    unsigned int q = nn->conv2d->parameters->topology[op][1];
    
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    
    static unsigned int local_idx;
    
    if (*advance > 0) {
        // Note that if local_idx is undefined because advance was incremented
        // by another op (pooling coming before convolution which is wrong), then
        // we get a non valid memory access.
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_matrices->rank; i++) {
            step = step * nn->conv2d->conv_matrices->shape[local_idx-1][i][0];
        }
        offset_m = offset_m + step;
        
        step = 1;
        for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
            step = step * nn->conv2d->conv_activations->shape[*advance-1][i][0];
        }
        offset_a = offset_a + step;
        
        step = 1;
        for (int i=0; i<nn->conv2d->conv_biases->rank; i++) {
            step = step * nn->conv2d->conv_biases->shape[local_idx-1][i][0];
        }
        offset_b = offset_b + step;
        
        step = 1;
        for (int i=0; i<nn->conv2d->conv_affineTransformations->rank; i++) {
            step = step * nn->conv2d->conv_affineTransformations->shape[local_idx-1][i][0];
        }
        offset_z = offset_z + step;
        
    } else {
        offset_m = 0;
        offset_a = 0;
        offset_a_compute = 0;
        offset_b = 0;
        offset_z = 0;
        offset_mask = 0;
        mask_idx = 0;
        local_idx = 0;
    }
    
    // Offset to activations at current convolution layer
    int step = 1;
    for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
        step = step * nn->conv2d->conv_activations->shape[*advance][i][0];
    }
    offset_a_compute = offset_a_compute + step;
    
    int rows_a = nn->conv2d->conv_activations->shape[*advance][1][0];
    int cols_a = nn->conv2d->conv_activations->shape[*advance][2][0];
    
    int rows_m = nn->conv2d->conv_matrices->shape[local_idx][2][0];
    int cols_m = nn->conv2d->conv_matrices->shape[local_idx][3][0];
    
    float vector[fh*fw];
    
    int stride2_m = 0;
    int stride_a_compute = 0;
    int stride_z = 0;
    for (int k=0; k<q; k++) { // Loop over all feature maps at current layer
        int stride_a = 0;
        int stride1_m = 0;
        memset(vector, 0.0f, sizeof(vector));
        for (int l=0; l<p; l++) { // Loop over all feature maps in previous layer
            
            cblas_sgemv(CblasRowMajor, CblasNoTrans, rows_m, cols_m, 1.0f, nn->conv2d->conv_matrices->val+offset_m+stride1_m+stride2_m, cols_m, nn->conv2d->conv_activations->val+offset_a+stride_a, 1, 1.0f, vector, 1);
            
            stride1_m = stride1_m + (nn->conv2d->conv_matrices->shape[local_idx][1][0] * cols_m * rows_m);
            stride_a = stride_a + (rows_a * cols_a);
        }
        
#ifdef __APPLE__
        vDSP_vsadd(vector, 1, nn->conv2d->conv_biases->val
                   +offset_b+k, nn->conv2d->conv_affineTransformations->val+offset_z+stride_z, 1, (fh*fw));
#else
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                nn->conv2d->conv_affineTransformations->val[offset_z+(stride_z+(i*fw+j))] = vector[i*fw+j] + nn->conv2d->conv_biases=>val[offset_b+k];
            }
        }
#endif
        
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*fw)+j))] = nn->conv2d->activationFunctions[local_idx](nn->conv2d->conv_affineTransformations->val[offset_z+(stride_z+(i*fw+j))], NULL, NULL);
            }
        }
        
        stride2_m = stride2_m + (rows_m * cols_m);
        stride_a_compute = stride_a_compute + (fh * fw);
        stride_z = stride_z + (fh * fw);
    }
    
    local_idx++;
}

void max_pooling_op(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int q = nn->conv2d->parameters->topology[op][1];
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    unsigned int kh = nn->conv2d->parameters->topology[op][4];
    unsigned int kw = nn->conv2d->parameters->topology[op][5];
    unsigned int sh = nn->conv2d->parameters->topology[op][6];
    unsigned int sw = nn->conv2d->parameters->topology[op][7];
    
    if (*advance > 0) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
            step = step * nn->conv2d->conv_activations->shape[*advance-1][i][0];
        }
        offset_a = offset_a + step;
        
        if (mask_idx > 0) {
            step = 1;
            for (int i=0; i<nn->conv2d->max_pool_mask->rank; i++) {
                step = step * nn->conv2d->max_pool_mask->shape[mask_idx-1][i][0];
            }
            offset_mask = offset_mask + step;
        }
        mask_idx++;
        
    } else {
        // A pooling op should not come so early int the network stack, but we allow
        // it only for being able to test the pooling operation independently.
        // We issue a warning to
        fprintf(stdout, "topology error in convolutional network. Pooling layer operation coming too early in the network operations stack.");
        offset_m = 0;
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
    
    int rows_a = nn->conv2d->conv_activations->shape[*advance][1][0];
    int cols_a = nn->conv2d->conv_activations->shape[*advance][2][0];
    
    int stride_a_compute = 0;
    int stride_a = 0;
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                float max_val = -HUGE_VALF;
                int winning_unit = 0;
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        if (nn->conv2d->conv_activations->val[offset_a+(stride_a+(((i*sh+u)*cols_a)+(j*sw+v)))] > max_val) {
                            max_val = nn->conv2d->conv_activations->val[offset_a+(stride_a+(((i*sh+u)*cols_a)+(j*sw+v)))];
                            winning_unit = stride_a+(((i*sh+u)*cols_a)+(j*sw+v));
                        }
                    }
                }
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*fw)+j))] = max_val;
                nn->conv2d->max_pool_mask->val[offset_mask+winning_unit] = 1.0f;
            }
        }
        stride_a = stride_a + (rows_a * cols_a);
        stride_a_compute = stride_a_compute + (fh * fw);
    }
}

void l2_pooling_op(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int q = nn->conv2d->parameters->topology[op][1];
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    unsigned int kh = nn->conv2d->parameters->topology[op][4];
    unsigned int kw = nn->conv2d->parameters->topology[op][5];
    unsigned int sh = nn->conv2d->parameters->topology[op][6];
    unsigned int sw = nn->conv2d->parameters->topology[op][7];
    
    if (*advance > 0) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
            step = step * nn->conv2d->conv_activations->shape[*advance-1][i][0];
        }
        offset_a = offset_a + step;
    } else {
        // A pooling op should not come so early int the network stack, but we allow
        // it only for being able to test the pooling operation independently.
        // We issue a warning to
        fprintf(stdout, "topology error in convolutional network. Pooling layer operation coming too early in the network operations stack.");
        offset_m = 0;
        offset_a = 0;
        offset_a_compute = 0;
        offset_b = 0;
    }
    
    int step = 1;
    for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
        step = step * nn->conv2d->conv_activations->shape[*advance][i][0];
    }
    offset_a_compute = offset_a_compute + step;
    
    int rows_a = nn->conv2d->conv_activations->shape[*advance][1][0];
    int cols_a = nn->conv2d->conv_activations->shape[*advance][2][0];
    
    int stride_a_compute = 0;
    int stride_a = 0;
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                float sum = 0.0f;
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        sum = sum + (nn->conv2d->conv_activations->val[offset_a+(stride_a+(((i*sh+u)*cols_a)+(j*sw+v)))] * nn->conv2d->conv_activations->val[offset_a+(stride_a+(((i*sh+u)*cols_a)+(j*sw+v)))]);
                    }
                }
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*fw)+j))] = sqrtf(sum);
            }
        }
        stride_a = stride_a + (rows_a * cols_a);
        stride_a_compute = stride_a_compute + (fh * fw);
    }
}

void average_pooling_op(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int q = nn->conv2d->parameters->topology[op][1];
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    unsigned int kh = nn->conv2d->parameters->topology[op][4];
    unsigned int kw = nn->conv2d->parameters->topology[op][5];
    unsigned int sh = nn->conv2d->parameters->topology[op][6];
    unsigned int sw = nn->conv2d->parameters->topology[op][7];
    
    if (*advance > 0) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
            step = step * nn->conv2d->conv_activations->shape[*advance-1][i][0];
        }
        offset_a = offset_a + step;
    } else {
        // A pooling op should not come so early int the network stack, but we allow
        // it only for being able to test the pooling operation independently.
        // We issue a warning to
        fprintf(stdout, "topology error in convolutional network. Pooling layer operation coming too early in the network operations stack.");
        offset_m = 0;
        offset_a = 0;
        offset_a_compute = 0;
        offset_b = 0;
    }
    
    int step = 1;
    for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
        step = step * nn->conv2d->conv_activations->shape[*advance][i][0];
    }
    offset_a_compute = offset_a_compute + step;
    
    int rows_a = nn->conv2d->conv_activations->shape[*advance][1][0];
    int cols_a = nn->conv2d->conv_activations->shape[*advance][2][0];
    
    int stride_a_compute = 0;
    int stride_a = 0;
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                float sum = 0.0f;
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        sum = sum +
                        nn->conv2d->conv_activations->val[offset_a+(stride_a+(((i*sh+u)*cols_a)+(j*sw+v)))];
                    }
                }
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+((i*fw)+j))] = sum / (kh*kw);
            }
        }
        stride_a = stride_a + (rows_a * cols_a);
        stride_a_compute = stride_a_compute + (fh * fw);
    }
}

void infer_fully_connected_op(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance) {
    
    static bool advance_connected_a;
    static bool advance_connected_compute;
    static int local_idx;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    tensor *ptr_activations = NULL;
    unsigned int *ptr_offset = NULL;
    
    if (*advance > 0) {
        if (nn->conv2d->parameters->topology[op-1][0] == POOLING ||
            nn->conv2d->parameters->topology[op-1][0] == CONVOLUTION) {
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
    
    unsigned int m = nn->conv2d->dense_weights->shape[local_idx][0][0];
    unsigned int n = nn->conv2d->dense_weights->shape[local_idx][1][0];
    
    float buffer[m];
    memset(buffer, 0.0f, sizeof(buffer));
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n, 1.0f, nn->conv2d->dense_weights->val+offset_w_connected, (int)n, ptr_activations->val+*ptr_offset, 1, 0.0f, buffer, 1);
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
    
    unsigned int advance = 0;
    for (int l=1; l<=nn->conv2d->num_infer_ops; l++) {
        nn->conv2d->inferenceOps[l-1](neural, l, &advance);
        advance++;
    }
}

//
// This routine does the backward propagation on the fully connected layers
// similarly to the routine used for a fully connected network.
//
void backpropag_full_connected_op(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance1, unsigned int * _Nullable advance2, unsigned int  * _Nullable advance3) {
    
    extern float * propag_delta;
    static unsigned int local_idx;
    
    tensor *ptr_activations = NULL;
    unsigned int *ptr_offset = NULL;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (*advance1 > 0) {
        if (nn->conv2d->parameters->topology[op-1][0] == FULLY_CONNECTED) {
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
        if (nn->conv2d->parameters->topology[op-1][0] == FULLY_CONNECTED) {
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
            // Stride to weights and dc_dw at last layer of fully connected part
            offset_w_connected = 0;
            offset_dcdw_connected = 0;
            for (int l=0; l<nn->conv2d->num_dense_layers-1; l++) {
                offset_w_connected = offset_w_connected + (nn->conv2d->dense_weights->shape[l][0][0]*nn->conv2d->dense_weights->shape[l][1][0]);
            }
            offset_dcdw_connected = offset_w_connected;
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
            
            offset_w_connected = 0;
            offset_a_connected = 0;
            offset_z_connected = 0;
            offset_dcdb_connected = 0;
            offset_dcdw_connected = 0;
        }
        
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
    
    if (op == nn->network_num_layers - 1) { // Last layer of the network
        
        // Compute delta
        int k = (int)nn->num_channels;
        for (int i=0; i<nn->conv2d->dense_activations->shape[nn->conv2d->num_dense_layers-1][0][0]; i++) {
            propag_delta[i] = nn->conv2d->dense_activations->val[offset_a_connected+i] - nn->batch[nn->example_idx][k];
            k++;
        }
        
        if (nn->conv2d->parameters->topology[op-1][0] == FULLY_CONNECTED) {
            offset_a_connected = offset_a_connected - nn->conv2d->dense_activations->shape[nn->conv2d->num_dense_layers-2][0][0];
            ptr_offset = &offset_a_connected;
        }
        
        int m = nn->conv2d->dense_batchCostWeightDeriv->shape[nn->conv2d->num_dense_layers-1][0][0];
        int n = nn->conv2d->dense_batchCostWeightDeriv->shape[nn->conv2d->num_dense_layers-1][1][0];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->conv2d->dense_batchCostWeightDeriv->val[offset_dcdw_connected+((i*n)+j)] = ptr_activations->val[*ptr_offset+j] * propag_delta[i];
            }
        }
        memcpy(nn->conv2d->dense_batchCostBiasDeriv->val+offset_dcdb_connected, propag_delta, nn->conv2d->dense_batchCostBiasDeriv->shape[nn->conv2d->num_dense_layers-1][0][0]*sizeof(float));

    } else { // Otherwise the layers up the fully connected part
        
        float buffer[nn->conv2d->parameters->max_number_nodes_in_dense_layer];
        memset(buffer, 0.0f, sizeof(buffer));
        
        float sp[nn->conv2d->dense_affineTransformations->shape[local_idx][0][0]];
        for (int i=0; i<nn->conv2d->dense_affineTransformations->shape[local_idx][0][0]; i++) {
            sp[i] = nn->conv2d->activationDerivatives[activ_idx](nn->conv2d->dense_affineTransformations->val[offset_z_connected+i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->conv2d->dense_weights->shape[local_idx+1][0][0], (int)nn->conv2d->dense_weights->shape[local_idx+1][1][0], 1.0, nn->conv2d->dense_weights->val+offset_w_connected, (int)nn->conv2d->dense_weights->shape[local_idx+1][1][0], propag_delta, 1, 0.0, buffer, 1);
        
#ifdef __APPLE__
        vDSP_vmul(buffer, 1, sp, 1, propag_delta, 1, nn->conv2d->dense_affineTransformations->shape[local_idx][0][0]);
#else
        for (int i=0; i<nn->conv2d->dense_affineTransformations->shape[local_idx][0][0]; i++) {
            global_buffer[i] = buffer[i] * sp[i];
        }
#endif
        int m = nn->conv2d->dense_batchCostWeightDeriv->shape[local_idx][0][0];
        int n = nn->conv2d->dense_batchCostWeightDeriv->shape[local_idx][1][0];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->conv2d->dense_batchCostWeightDeriv->val[offset_dcdw_connected+((i*n)+j)] = ptr_activations->val[*ptr_offset+j] * propag_delta[i];
            }
        }
        
        memcpy(nn->conv2d->dense_batchCostBiasDeriv->val+offset_dcdb_connected, propag_delta, nn->conv2d->dense_batchCostBiasDeriv->shape[local_idx][0][0]*sizeof(float));
        
        offset_w_connected = offset_w_connected - (nn->conv2d->dense_weights->shape[local_idx][0][0] * nn->conv2d->dense_weights->shape[local_idx][1][0]);
    }
    
    activ_idx--;
}

//
// This routine computes the dela_{l} at the current convolution layer and updates
// the derivatives of the cost function with respect to the convolution weights and biases
//
void backpropag_convolution_op(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance1, unsigned int * _Nullable advance2, unsigned int  * _Nullable advance3) {
    
    extern float * propag_delta;
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int p = nn->conv2d->parameters->topology[op-1][1];
    unsigned int q = nn->conv2d->parameters->topology[op][1];
    
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    unsigned int fh_p = nn->conv2d->parameters->topology[op-1][2];
    unsigned int fw_p = nn->conv2d->parameters->topology[op-1][3];
    unsigned int kh = nn->conv2d->parameters->topology[op][4];
    unsigned int kw = nn->conv2d->parameters->topology[op][5];
    unsigned int sh = nn->conv2d->parameters->topology[op][6];
    unsigned int sw = nn->conv2d->parameters->topology[op][7];
    
    int offset_z = 0;
    for (int l=0; l<*advance2; l++) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_affineTransformations->rank; i++) {
            step = step * nn->conv2d->conv_affineTransformations->shape[l][i][0];
        }
        offset_z = offset_z + step;
    }
    
    // The term delta_{l}
    if (nn->conv2d->parameters->topology[op+1][0] == POOLING) {
        // We get the term delta^{l+1}*rot{k^{l+1}} or transpose(w^{l+1}) x delta^{l+1}
        // upsampled from a pooling layer
        int stride = 0;
        for (int k=0; k<q; k++) {
            for (int i=0; i<fh; i++) {
                for (int j=0; j<fw; j++) {
                    nn->conv2d->propag_upsampling[stride+(i*fw+j)] = nn->conv2d->propag_upsampling[stride+(i*fw+j)] * nn->conv2d->activationDerivatives[activ_idx](nn->conv2d->conv_affineTransformations->val[offset_z+(stride+(i*fw+j))]);
                }
            }
            stride = stride + (fh * fw);
        }
        
    } else if (nn->conv2d->parameters->topology[op+1][0] == CONVOLUTION) {
        // We get delta_{l+1} and compute here delta^{l+1}*rot{k^{l+1}}
        unsigned int offset_m = 0;
        for (int l=0; l<=*advance2; l++) {
            int step = 1;
            for (int i=0; i<nn->conv2d->conv_matrices->rank; i++) {
                step = step * nn->conv2d->conv_matrices->shape[l][i][0];
            }
            offset_m = offset_m + step;
        }
        
        float vector[fh*fw];
        unsigned int p = nn->conv2d->parameters->topology[op][1];
        unsigned int q = nn->conv2d->parameters->topology[op+1][1];
        int rows_m = nn->conv2d->conv_matrices->shape[*advance2+1][2][0];
        int cols_m = nn->conv2d->conv_matrices->shape[*advance2+1][3][0];
        int rows_d = nn->conv2d->parameters->topology[op+1][2];
        int cols_d = nn->conv2d->parameters->topology[op+1][3];
        
        int stride1_m = 0;
        int stride_s = 0;
        for (int k=0; k<p; k++) {
            int stride_d = 0;
            int stride2_m = 0;
            memset(vector, 0.0f, sizeof(vector));
            for (int l=0; l<q; l++) {
                cblas_sgemv(CblasRowMajor, CblasTrans, rows_m, cols_m, 1.0f, nn->conv2d->conv_matrices->val+offset_m+stride1_m+stride2_m, cols_m, nn->conv2d->propag_upsampling+stride_d, 1, 1.0, vector, 1);
                stride2_m = stride2_m + (rows_m * cols_m);
                stride_d = stride_d + (rows_d * cols_d);
            }
            memcpy(propag_delta+stride_s, vector, (fh*fw)*sizeof(float));
            stride1_m = stride1_m + (nn->conv2d->conv_matrices->shape[*advance2+1][1][0] * rows_m * cols_m);
            stride_s = stride_s + (fh * fw);
        }
        
        int stride = 0;
        for (int k=0; k<p; k++) {
            for (int i=0; i<fh; i++) {
                for (int j=0; j<fw; j++) {
                    nn->conv2d->propag_upsampling[stride+(i*fw+j)] = propag_delta[stride+(i*fw+j)] * nn->conv2d->activationDerivatives[activ_idx](nn->conv2d->conv_affineTransformations->val[offset_z+(stride+(i*fw+j))]);
                }
            }
            stride = stride + (fh * fw);
        }
    }
    
    // Flip the deltas (errors)
    nn->flip_deltas((void *)nn, q, fh, fw);
    
    // Offset to activations at previous layer
    int offset_a = 0;
    for (int l=0; l<op-1; l++) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_activations->rank; i++) {
            step = step * nn->conv2d->conv_activations->shape[l][i][0];
        }
        offset_a = offset_a + step;
    }
    
    // Offset to dC/dw
    int offset_w = 0;
    for (int l=0; l<*advance2; l++) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_batchCostWeightDeriv->rank; i++) {
            step = step * nn->conv2d->conv_batchCostWeightDeriv->shape[l][i][0];
        }
        offset_w = offset_w + step;
    }
    
    // Offset to dC/db
    int offset_b = 0;
    for (int l=0; l<*advance2; l++) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_batchCostBiasDeriv->rank; i++) {
            step = step * nn->conv2d->conv_batchCostBiasDeriv->shape[l][i][0];
        }
        offset_b = offset_b + step;
    }
    
    // We update the dC/dw by computing the convolution operation.
    // TODO: Try to replace the convolution with the more efficient Matrix-Vector
    // multiplication operation
    
    int stride_a = 0;
    int stride1_m = 0;
    for (int k=0; k<p; k++) {
        int stride = 0;
        int stride2_m = 0;
        for (int l=0; l<q; l++) {
            for (int u=0; u<kh; u++) {
                for (int v=0; v<kw; v++) {
                    float sum_w = 0.0f;
                    for (int i=0; i<fh; i++) {
                        for (int j=0; j<fw; j++) {
                            sum_w = sum_w + propag_delta[stride+(i*fw+j)] * nn->conv2d->conv_activations->val[offset_a+(stride_a+((i*sh+u)*fw_p+(j*sw+v)))];
                        }
                    }
                    nn->conv2d->conv_batchCostWeightDeriv->val[offset_w+(stride1_m+(stride2_m+(u*kw+v)))] = sum_w;
                }
            }
            stride = stride + (fh * fw);
            stride2_m = stride2_m + (kh * kw);
        }
        stride_a = stride_a + (fh_p * fw_p);
        stride1_m = stride1_m + (q * kh * kw);
    }
    
    int stride = 0;
    for (int l=0; l<q; l++) {
        float sum_b = 0.0f;
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                sum_b = sum_b + nn->conv2d->propag_upsampling[stride+(i*fw+j)];
            }
        }
        nn->conv2d->conv_batchCostBiasDeriv->val[offset_b+l] = sum_b;
        stride = stride + (fh * fw);
    }
    
    (*advance2)--;
    activ_idx--;
}

//
// This routine is called by the backpropagation pooling op when the pooling layer follows a dense layer
// going the network upward.
// It computes the term transpose(w^{l+1}) x delta^{l+1}, where w^{l+1} and delta^{l+1} are the weights
// and errors of the dense layer which follows the pooling layer downward.
//
static void backpropag_pooling_after_fully_connected(void * _Nonnull neural, unsigned int op) {
    
    extern float * propag_delta;
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    int length = nn->conv2d->parameters->topology[op][1]*nn->conv2d->parameters->topology[op][2] *
    nn->conv2d->parameters->topology[op][3];
    
    float buffer[length];
    memset(buffer, 0.0f, sizeof(buffer));
    
    cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->conv2d->dense_weights->shape[0][0][0], (int)nn->conv2d->dense_weights->shape[0][1][0], 1.0, nn->conv2d->dense_weights->val, (int)nn->conv2d->dense_weights->shape[0][1][0], propag_delta, 1, 0.0, buffer, 1);
    memcpy(propag_delta, buffer, length*sizeof(float));
}

//
// This routine is called by the backpropagation pooling op when the pooling layer follows a convolution
// layer crossing the network upward.
// It computes the term delta^{l+1}*rot{k^{l+1}}, where * is the convolution operation, delta^{l+1} and
// k{l+1} are the error (stored in the propag_upsampling array) and the weights of the convolution layer which
// follows the pooling layer downward.
//
static void backpropag_pooling_after_convolution(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance2) {
    
    extern float * propag_delta;
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int p = nn->conv2d->parameters->topology[op][1];
    unsigned int q = nn->conv2d->parameters->topology[op+1][1];
    
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    
    unsigned int offset_m = 0;
    for (int l=0; l<=*advance2; l++) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_matrices->rank; i++) {
            step = step * nn->conv2d->conv_matrices->shape[l][i][0];
        }
        offset_m = offset_m + step;
    }
    
    int rows_m = nn->conv2d->conv_matrices->shape[*advance2+1][2][0];
    int cols_m = nn->conv2d->conv_matrices->shape[*advance2+1][3][0];
    
    int rows_d = nn->conv2d->parameters->topology[op+1][2];
    int cols_d = nn->conv2d->parameters->topology[op+1][3];
    
    float vector[fh*fw];
   
    int stride1_m = 0;
    int stride_s = 0;
    for (int k=0; k<p; k++) { // Loop accross maps in pooling layer
        int stride_d = 0;
        int stride2_m = 0;
        memset(vector, 0.0f, sizeof(vector));
        for (int l=0; l<q; l++) {
            cblas_sgemv(CblasRowMajor, CblasTrans, rows_m, cols_m, 1.0f, nn->conv2d->conv_matrices->val+offset_m+stride1_m+stride2_m, cols_m, nn->conv2d->propag_upsampling+stride_d, 1, 1.0, vector, 1);
            stride2_m = stride2_m + (rows_m * cols_m);
            stride_d = stride_d + (rows_d * cols_d);
        }
        memcpy(propag_delta+stride_s, vector, (fh*fw)*sizeof(float));
        stride1_m = stride1_m + (nn->conv2d->conv_matrices->shape[*advance2+1][1][0] * rows_m * cols_m);
        stride_s = stride_s + (fh * fw);
    }
}

void backpropag_max_pooling_op(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance1, unsigned int * _Nullable advance2, unsigned int  * _Nullable advance3) {
    
    extern float * propag_delta;
    static bool check = false;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (!check) {
        if (*advance1 == 0) fatal(DEFAULT_CONSOLE_WRITER, "topology error in convolutional network. Using a pooling layer as an output layer is not permitted.");
        check = true;
    }
    
    int offset = 0;
    for (int l=0; l<*advance3; l++) {
        int step = 1;
        for (int i=0; i<nn->conv2d->max_pool_mask->rank; i++) {
            step = step * nn->conv2d->max_pool_mask->shape[l][i][0];
        }
        offset = offset + step;
    }
    
    if (nn->conv2d->parameters->topology[op+1][0] == CONVOLUTION) {
        backpropag_pooling_after_convolution((void *)nn, op, advance2);
    } else {
        backpropag_pooling_after_fully_connected((void *)nn, op);
    }
    
    // Upsampling: The error is directly assigned to the winning unit
    // kh and kw are the horizontal and vertical dimension of the pooling kernel respectively
    unsigned int q = nn->conv2d->parameters->topology[op-1][1];
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    unsigned int kh = nn->conv2d->parameters->topology[op][4];
    unsigned int kw = nn->conv2d->parameters->topology[op][5];
    unsigned int sh = nn->conv2d->parameters->topology[op][6];
    unsigned int sw = nn->conv2d->parameters->topology[op][7];
    
    unsigned int cols_c = nn->conv2d->parameters->topology[op-1][3];
    unsigned int cols_s = nn->conv2d->parameters->topology[op][3];
    
    int stride_c = 0;
    int stride_s = 0;
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        nn->conv2d->propag_upsampling[stride_c+(((i*sh+u)*cols_c)+(j*sw+v))] =
                        propag_delta[stride_s+((i*cols_s)+j)]*nn->conv2d->max_pool_mask->val[offset+(stride_c+(((i*sh+u)*cols_c)+(j*sw+v)))];
                    }
                }
            }
        }
        stride_c = stride_c + (nn->conv2d->parameters->topology[op-1][2] * nn->conv2d->parameters->topology[op-1][3]);
        stride_s = stride_s + (nn->conv2d->parameters->topology[op][2] * nn->conv2d->parameters->topology[op][3]);
    }
    
    (*advance3)--;
}

void backpropag_l2_pooling_op(void * _Nonnull neural, unsigned int op, unsigned int *_Nullable advance1, unsigned int * _Nullable advance2, unsigned int  * _Nullable advance3) {
    
}

void backpropag_average_pooling_op(void * _Nonnull neural, unsigned int op, unsigned int * _Nullable advance1, unsigned int * _Nullable advance2, unsigned int  * _Nullable advance3) {
    
    extern float * propag_delta;
    static bool check = false;
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    if (!check) {
        if (*advance1 == 0) fatal(DEFAULT_CONSOLE_WRITER, "topology error in convolutional network. Using a pooling layer as an output layer is not permitted.");
        check = true;
    }
    
    if (nn->conv2d->parameters->topology[op+1][0] == CONVOLUTION) {
        backpropag_pooling_after_convolution((void *)nn, op, advance2);
    } else {
        backpropag_pooling_after_fully_connected((void *)nn, op);
    }
    
    // Upsampling: the error is multiplied by 1/(kh*kw) and assigned to the
    // whole pooling block. kh and kw are the horizontal and vertical dimension
    // of the pooling kernel respectively
    unsigned int q = nn->conv2d->parameters->topology[op-1][1];
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    unsigned int kh = nn->conv2d->parameters->topology[op][4];
    unsigned int kw = nn->conv2d->parameters->topology[op][5];
    unsigned int sh = nn->conv2d->parameters->topology[op][6];
    unsigned int sw = nn->conv2d->parameters->topology[op][7];
    
    unsigned int cols_c = nn->conv2d->parameters->topology[op-1][3];
    unsigned int cols_s = nn->conv2d->parameters->topology[op][3];
    
    int stride_c = 0;
    int stride_s = 0;
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        nn->conv2d->propag_upsampling[stride_c+(((i*sh+u)*cols_c)+(j*sw+v))] =
                                1.0/((float)(kh*kw))*propag_delta[stride_s+((i*cols_s)+j)];
                    }
                }
            }
        }
        stride_c = stride_c + (nn->conv2d->parameters->topology[op-1][2] * nn->conv2d->parameters->topology[op-1][3]);
        stride_s = stride_s + (nn->conv2d->parameters->topology[op][2] * nn->conv2d->parameters->topology[op][3]);
    }
}

void backpropag_in_conv2d_net(void * _Nonnull neural,
                              void (* _Nullable ptr_inference_func)(void * _Nonnull self)) {
    
    static bool first_time = true;
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Compute only once the length of the tensor storing the max pooling mask
    static int length = 0;
    if (first_time) {
        for (int l=0; l<nn->conv2d->num_max_pooling_layers; l++) {
            int size = 1;
            for (int i=0; i<nn->conv2d->max_pool_mask->rank; i++) {
                size = size * nn->conv2d->max_pool_mask->shape[l][i][0];
            }
            length = length + size;
        }
        first_time = false;
    }
    
    // Activations at the input layer
    // TODO: it would be maybe better to have the batch being
    // defined as a 4D tensor of shape [mini-batch size, height, width, channels]
    // Note that inputs with multi-channels are not supported yet
    for (int i=0; i<nn->num_channels; i++) {
        nn->conv2d->conv_activations->val[i] = nn->batch[nn->example_idx][i];
    }
    
    // Inference (forward pass)
    memset(nn->conv2d->max_pool_mask->val, 0.0f, length*sizeof(float));
    ptr_inference_func(neural);
    
    // Backpropagation
    unsigned int advance1 = 0;
    unsigned int advance2 = nn->conv2d->num_conv2d_layers - 1;
    unsigned int advance3 = nn->conv2d->num_max_pooling_layers - 1;
    for (int l=nn->conv2d->num_backpropag_ops; l>=1; l--) {
        nn->conv2d->backpropagOps[l-1](neural, l, &advance1, &advance2, &advance3);
        advance1++;
    }
}

void batch_accumulation_in_conv2d_net(void * _Nonnull neural) {
    
    // Accumulate dC/dw and dC/db at convolution and
    // fully connected layers
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Convolution layers
    int offset_w = 0;
    int offset_b = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned int p = nn->conv2d->conv_batchCostWeightDeriv->shape[l][0][0];
        unsigned int q = nn->conv2d->conv_batchCostWeightDeriv->shape[l][1][0];
        unsigned int kh = nn->conv2d->conv_batchCostWeightDeriv->shape[l][2][0];
        unsigned int kw = nn->conv2d->conv_batchCostWeightDeriv->shape[l][3][0];
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        nn->conv2d->conv_costWeightDerivatives->val[offset_w+(stride1+(stride2+(u*kw+v)))] =
                          nn->conv2d->conv_costWeightDerivatives->val[offset_w+(stride1+(stride2+(u*kw+v)))] +
                              nn->conv2d->conv_batchCostWeightDeriv->val[offset_w+(stride1+(stride2+(u*kw+v)))];
                    }
                }
                stride2 = stride2 + (kh * kw);
            }
            stride1 = stride1 + (q * kh * kw);
        }
        
        for (int ll=0; ll<q; ll++) {
            nn->conv2d->conv_costBiasDerivatives->val[offset_b+ll] = nn->conv2d->conv_costBiasDerivatives->val[offset_b+ll] + nn->conv2d->conv_batchCostBiasDeriv->val[offset_b+ll];
        }
        
        offset_w = offset_w + (p * q * kh * kw);
        offset_b = offset_b + q;
    }
    
    // Fully connected layers
    offset_w = 0;
    offset_b = 0;
    for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
        unsigned int m = nn->conv2d->dense_costWeightDerivatives->shape[l][0][0];
        unsigned int n = nn->conv2d->dense_costWeightDerivatives->shape[l][1][0];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->conv2d->dense_costWeightDerivatives->val[offset_w+((i*n)+j)] =
                   nn->conv2d->dense_costWeightDerivatives->val[offset_w+((i*n)+j)] + nn->conv2d->dense_batchCostWeightDeriv->val[offset_w+((i*n)+j)];
            }
        }
        for (int i=0; i<m; i++) {
            nn->conv2d->dense_costBiasDerivatives->val[offset_b+i] =
            nn->conv2d->dense_costBiasDerivatives->val[offset_b+i] + nn->conv2d->dense_batchCostBiasDeriv->val[offset_b+i];
        }
        
        offset_w = offset_w + (m * n);
        offset_b = offset_b + m;
    }
}
