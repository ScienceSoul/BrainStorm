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

static unsigned int offset_km;
static unsigned int offset_a;
static unsigned int offset_a_compute;
static unsigned int offset_b;
static unsigned int offset_z;
static unsigned int offset_max_pool_index_store;
static unsigned int store_idx;

static unsigned int offset_a_connected;
static unsigned int offset_a_connected_compute;
static unsigned int offset_w_connected;
static unsigned int offset_b_connected;
static unsigned int offset_z_connected;
static unsigned int offset_dcdw_connected;
static unsigned int offset_dcdb_connected;

static unsigned int activ_idx;

static void update_conv_input(void * _Nonnull neural, unsigned int offset, unsigned int p, unsigned int fh, unsigned int fw, unsigned int ld1, unsigned int ld2, unsigned int kh, unsigned int kw, unsigned int sh, unsigned int sw) {
    
    extern tensor *conv_input_matrix;
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    int indx = 0;
    for (int i=0; i<fh; i++) {
        for (int j=0; j<fw; j++) {
            int stride_a = 0;
            for (int ll=0; ll<p; ll++) {
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        conv_input_matrix->val[indx] = nn->conv2d->conv_activations->val[offset+(stride_a+(((i*sh+u)*ld2)+(j*sw+v)))];
                        indx++;
                    }
                }
                stride_a = stride_a + (ld1 * ld2);
            }
        }
    }
}

// -----------------------------------------------------------------------
// ---- Unrolled back-propagation.
// ---- nabla_X = nabla_Y * trans(W) (Chellapilla et al, 2006)
// -----------------------------------------------------------------------
static void transpose_convolution_op(void * _Nonnull neural, unsigned int op,  int * _Nullable advance2) {
    
    extern tensor * propag_buffer;
    extern tensor *conv_input_matrix;
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    unsigned int p = nn->conv2d->parameters->topology[op][1];
    unsigned int q = nn->conv2d->parameters->topology[op+1][1];
    
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    
    
    unsigned int offset = 0;
    for (int l=0; l<=*advance2; l++) {
        int step = 1;
        for (int i=0; i<nn->conv2d->kernel_matrices->rank; i++) {
            step = step * nn->conv2d->kernel_matrices->shape[l][i][0];
        }
        offset = offset + step;
    }
    
    int rows_d = nn->conv2d->parameters->topology[op+1][2];
    int cols_d = nn->conv2d->parameters->topology[op+1][3];
    
    // The matrix-matrix product implementation of the transpose convolution
    float A[(rows_d*cols_d)*q];
    int stride_d = 0;
    for (int l=0; l<q; l++) {
        int indx = 0;
        for (int i=0; i<rows_d; i++) {
            for (int j=0; j<cols_d; j++) {
                A[indx*q+l] = nn->conv2d->deltas_buffer->val[stride_d+(i*cols_d+j)];
                indx++;
            }
        }
        stride_d = stride_d + (rows_d * cols_d);
    }
    
    unsigned int kh = nn->conv2d->parameters->topology[op+1][4];
    unsigned int kw = nn->conv2d->parameters->topology[op+1][5];
    unsigned int sh =  nn->conv2d->parameters->topology[op+1][6];
    unsigned int sw =  nn->conv2d->parameters->topology[op+1][7];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows_d*cols_d, p*kh*kw, q, 1.0f, A, q, nn->conv2d->kernel_matrices->val+offset, q, 0.0f, conv_input_matrix->val, p*kh*kw);
    
    // Recover each feature maps with their deltas
    memset(propag_buffer->val, 0.0f, propag_buffer->shape[0][0][0]*sizeof(float));
    int stride_s = 0;
    for (int k=0; k<p; k++) {
        int m = 0;
        for (int i=0; i<rows_d; i++) {
            for (int j=0; j<cols_d; j++) {
                int indx = 0;
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        propag_buffer->val[stride_s+(((i*sh+u)*fw)+(j*sw+v))] =
                        propag_buffer->val[stride_s+(((i*sh+u)*fw)+(j*sw+v))] + conv_input_matrix->val[(m*(p*kh*kw))+(kh*kw*k)+indx];
                        indx++;
                    }
                }
                m++;
            }
        }
        stride_s = stride_s + (fh * fw);
    }
}

void transpose_convolution(void * _Nonnull neural, unsigned int op,  int * _Nullable advance2) {
    transpose_convolution_op(neural, op, advance2);
}


// -----------------------------------------------------------------------
// ---- Forward-propagation. Processing in each convolutional layer using
// ---- unrolled convolution (matrix-patrix product)
// ---- Y = X * W (Chellapilla et al, 2006)
// -----------------------------------------------------------------------
void infer_convolution_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance) {
    
    extern tensor *conv_input_matrix;
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    unsigned int p = nn->conv2d->parameters->topology[op-1][1];
    unsigned int q = nn->conv2d->parameters->topology[op][1];
    
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    unsigned int kh = nn->conv2d->parameters->topology[op][4];
    unsigned int kw = nn->conv2d->parameters->topology[op][5];
    unsigned int sh = nn->conv2d->parameters->topology[op][6];
    unsigned int sw = nn->conv2d->parameters->topology[op][7];
    
    static unsigned int local_idx;
    
    if (*advance > 0) {
        // If local_idx is undefined because advance was already incremented
        // by another op (pooling coming before convolution which is wrong), then
        // we fatal
        if (local_idx <= 0) {
            fatal(DEFAULT_CONSOLE_WRITER, "topology error in convolutional network. A pooling layer is probably coming too early in the network operations stack.");
        }
        
        int step = 1;
        for (int i=0; i<nn->conv2d->kernel_matrices->rank; i++) {
            step = step * nn->conv2d->kernel_matrices->shape[local_idx-1][i][0];
        }
        offset_km = offset_km + step;
        
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
        for (int i=0; i<nn->conv2d->conv_affine_transforms->rank; i++) {
            step = step * nn->conv2d->conv_affine_transforms->shape[local_idx-1][i][0];
        }
        offset_z = offset_z + step;
        
    } else {
        offset_km = 0;
        offset_a = 0;
        offset_a_compute = 0;
        offset_b = 0;
        offset_z = 0;
        offset_max_pool_index_store = 0;
        store_idx = 0;
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
    
    // Update the input matrix for the convolution operation
    update_conv_input(neural, offset_a, p, fh, fw, rows_a, cols_a, kh, kw, sh, sw);
    
    float C[fh*fw][q];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, fh*fw, q, p*kh*kw, 1.0f, conv_input_matrix->val, p*kh*kw, nn->conv2d->kernel_matrices->val+offset_km, q, 0.0f, *C, q);

#ifdef __APPLE__
    float C_trans[q][fh*fw];
    vDSP_mtrans(*C, 1, *C_trans, 1, q, fh*fw);
#endif
    
    int stride_a_compute = 0;
    int stride_z = 0;
    for (int k=0; k<q; k++) {
#ifdef __APPLE__
        vDSP_vsadd(*C_trans+(k*fh*fw), 1, nn->conv2d->conv_biases->val
                   +offset_b+k, nn->conv2d->conv_affine_transforms->val+offset_z+stride_z, 1, (fh*fw));
#else
        int indx = 0;
        for (int i=0; i<fh*fw; i++) {
            nn->conv2d->conv_affineTransformations->val[offset_z+(stride_z+i)] = C[indx][k] + nn->conv2d->conv_biases->val[offset_b+k];
            indx++;
        }
#endif
        
        for (int i=0; i<fh*fw; i++) {
                nn->conv2d->conv_activations->val[offset_a_compute+(stride_a_compute+i)] = nn->conv2d->activation_functions[local_idx](nn->conv2d->conv_affine_transforms->val[offset_z+(stride_z+i)], NULL, NULL);
        }
        nanToNum(nn->conv2d->conv_activations->val+offset_a_compute+stride_a_compute, (fh*fw));
        
        stride_a_compute = stride_a_compute + (fh * fw);
        stride_z = stride_z + (fh * fw);
    }
    
    local_idx++;
}

void max_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
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
        
        if (store_idx > 0) {
            step = 1;
            for (int i=0; i<nn->conv2d->max_pool_indexes->rank; i++) {
                step = step * nn->conv2d->max_pool_indexes->shape[store_idx-1][i][0];
            }
            offset_max_pool_index_store = offset_max_pool_index_store + step;
        }
        
    } else {
        // A pooling op should not come so early in the network stack, but we allow
        // it only for being able to test the pooling operation independently.
        fprintf(stdout, "WARNING: topology error in convolutional network. Pooling layer operation coming too early in the network operations stack.\n");
        offset_km = 0;
        offset_a = 0;
        offset_a_compute = 0;
        offset_b = 0;
        offset_max_pool_index_store = 0;
        store_idx = 0;
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
                nn->conv2d->max_pool_indexes->int32_val[offset_max_pool_index_store+(stride_a_compute+((i*fw)+j))] = winning_unit;
            }
        }
        stride_a = stride_a + (rows_a * cols_a);
        stride_a_compute = stride_a_compute + (fh * fw);
    }
    store_idx++;
}

void l2_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
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
        // A pooling op should not come so early in the network stack, but we allow
        // it only for being able to test the pooling operation independently.
        fprintf(stdout, "WARNING: topology error in convolutional network. Pooling layer operation coming too early in the network operations stack.\n");
        offset_km = 0;
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

void average_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
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
        // A pooling op should not come so early in the network stack, but we allow
        // it only for being able to test the pooling operation independently.
        fprintf(stdout, "WARNING: topology error in convolutional network. Pooling layer operation coming too early in the network operations stack.\n");
        offset_km = 0;
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

void infer_fully_connected_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance) {
    
    static bool advance_connected_a;
    static bool advance_connected_compute;
    static int local_idx;
    
    brain_storm_net *nn = (brain_storm_net *)neural;
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
        offset_z_connected = offset_z_connected + nn->conv2d->dense_affine_transforms->shape[local_idx-1][0][0];
    }
    
    unsigned int m = nn->conv2d->dense_weights->shape[local_idx][0][0];
    unsigned int n = nn->conv2d->dense_weights->shape[local_idx][1][0];
    
    float buffer[m];
    memset(buffer, 0.0f, sizeof(buffer));
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n, 1.0f, nn->conv2d->dense_weights->val+offset_w_connected, (int)n, ptr_activations->val+*ptr_offset, 1, 0.0f, buffer, 1);
#ifdef __APPLE__
    vDSP_vadd(buffer, 1, nn->conv2d->dense_biases->val+offset_b_connected, 1, nn->conv2d->dense_affine_transforms->val+offset_z_connected, 1, nn->conv2d->dense_biases->shape[local_idx][0][0]);
#else
    for (int i=0; i<nn->conv2d->dense_biases->shape[local_idx][0][0]; i++) {
        nn->conv2d->dense_affineTransformations->val[offset_z_connected+i] = buffer[i] + nn->conv2d->dense_biases->val[offset_b_connected+i];
    }
#endif
    
    float *vec = NULL;
    unsigned int *vec_length = NULL;
    // To get the activation function associated with a fully connected layer, we assume that
    // the fully connected layers always come after all convolution layers
    // (no interleaving between convolution layers) which should be the case for a correctly
    // constructed convolutional network. Otherwise wrong behavior and results!!
    if (nn->activation_functions_ref[local_idx+nn->conv2d->num_conv2d_layers] == SOFTMAX) {
        vec = nn->conv2d->dense_affine_transforms->val+offset_z_connected;
        vec_length = &(nn->conv2d->dense_affine_transforms->shape[local_idx][0][0]);
    }
    for (int i=0; i<nn->conv2d->dense_activations->shape[local_idx][0][0]; i++) {
        nn->conv2d->dense_activations->val[offset_a_connected_compute+i] =
             nn->conv2d->activation_functions[local_idx+nn->conv2d->num_conv2d_layers](nn->conv2d->dense_affine_transforms->val[offset_z_connected+i], vec, vec_length);
    }
    
    nanToNum(nn->conv2d->dense_activations->val+offset_a_connected_compute, nn->conv2d->dense_activations->shape[local_idx][0][0]);
}

void inference_in_conv2d_net(void * _Nonnull neural) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    int advance = 0;
    for (int l=1; l<=nn->conv2d->num_infer_ops; l++) {
        nn->conv2d->inference_ops[l-1](neural, l, &advance);
        advance++;
    }
}

//
// This routine does the backward propagation on the fully connected layers
// similarly to the routine used for a fully connected network.
//
void backpropag_full_connected_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance1, int * _Nullable advance2, int  * _Nullable advance3) {
    
    extern tensor * propag_buffer;
    static unsigned int local_idx;
    
    tensor *ptr_activations = NULL;
    unsigned int *ptr_offset = NULL;
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
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
        
        offset_z_connected = offset_z_connected - nn->conv2d->dense_affine_transforms->shape[local_idx-1][0][0];
        offset_dcdw_connected = offset_dcdw_connected - (nn->conv2d->dense_batch_cost_weight_derivs->shape[local_idx-1][0][0]*nn->conv2d->dense_batch_cost_weight_derivs->shape[local_idx-1][1][0]);
        offset_dcdb_connected = offset_dcdb_connected - nn->conv2d->dense_batch_cost_bias_derivs->shape[local_idx-1][0][0];
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
                offset_z_connected = offset_z_connected + nn->conv2d->dense_affine_transforms->shape[l][0][0];
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
    }
    
    if (op == nn->network_num_layers - 1) { // Last layer of the network
        
        // Compute delta
        int k = (int)nn->num_channels;
        for (int i=0; i<nn->conv2d->dense_activations->shape[nn->conv2d->num_dense_layers-1][0][0]; i++) {
            propag_buffer->val[i] = nn->conv2d->dense_activations->val[offset_a_connected+i] - nn->batch_labels->val[nn->label_step*nn->example_idx+i];
            k++;
        }
        
        if (nn->conv2d->parameters->topology[op-1][0] == FULLY_CONNECTED) {
            offset_a_connected = offset_a_connected - nn->conv2d->dense_activations->shape[nn->conv2d->num_dense_layers-2][0][0];
            ptr_offset = &offset_a_connected;
        }
        
        int m = nn->conv2d->dense_batch_cost_weight_derivs->shape[nn->conv2d->num_dense_layers-1][0][0];
        int n = nn->conv2d->dense_batch_cost_weight_derivs->shape[nn->conv2d->num_dense_layers-1][1][0];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->conv2d->dense_batch_cost_weight_derivs->val[offset_dcdw_connected+((i*n)+j)] = ptr_activations->val[*ptr_offset+j] * propag_buffer->val[i];
            }
        }
        memcpy(nn->conv2d->dense_batch_cost_bias_derivs->val+offset_dcdb_connected, propag_buffer->val, nn->conv2d->dense_batch_cost_bias_derivs->shape[nn->conv2d->num_dense_layers-1][0][0]*sizeof(float));

    } else { // Otherwise the layers up the fully connected part
        
        float buffer[nn->conv2d->parameters->max_number_nodes_in_dense_layer];
        memset(buffer, 0.0f, sizeof(buffer));
        
        float sp[nn->conv2d->dense_affine_transforms->shape[local_idx][0][0]];
        for (int i=0; i<nn->conv2d->dense_affine_transforms->shape[local_idx][0][0]; i++) {
            sp[i] = nn->conv2d->activation_derivatives[activ_idx](nn->conv2d->dense_affine_transforms->val[offset_z_connected+i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->conv2d->dense_weights->shape[local_idx+1][0][0], (int)nn->conv2d->dense_weights->shape[local_idx+1][1][0], 1.0, nn->conv2d->dense_weights->val+offset_w_connected, (int)nn->conv2d->dense_weights->shape[local_idx+1][1][0], propag_buffer->val, 1, 0.0, buffer, 1);
        
#ifdef __APPLE__
        vDSP_vmul(buffer, 1, sp, 1, propag_buffer->val, 1, nn->conv2d->dense_affine_transforms->shape[local_idx][0][0]);
#else
        for (int i=0; i<nn->conv2d->dense_affineTransformations->shape[local_idx][0][0]; i++) {
            propag_buffer->val[i] = buffer[i] * sp[i];
        }
#endif
        int m = nn->conv2d->dense_batch_cost_weight_derivs->shape[local_idx][0][0];
        int n = nn->conv2d->dense_batch_cost_weight_derivs->shape[local_idx][1][0];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->conv2d->dense_batch_cost_weight_derivs->val[offset_dcdw_connected+((i*n)+j)] = ptr_activations->val[*ptr_offset+j] * propag_buffer->val[i];
            }
        }
        
        memcpy(nn->conv2d->dense_batch_cost_bias_derivs->val+offset_dcdb_connected, propag_buffer->val, nn->conv2d->dense_batch_cost_bias_derivs->shape[local_idx][0][0]*sizeof(float));
        
        offset_w_connected = offset_w_connected - (nn->conv2d->dense_weights->shape[local_idx][0][0] * nn->conv2d->dense_weights->shape[local_idx][1][0]);
    }
    
    activ_idx--;
}

//
// This routine is used to propagate the error from a fully connected layer going upward in the network.
// It computes the term transpose(w^{l+1}) x delta^{l+1}, where w^{l+1} and delta^{l+1} are the weights
// and errors of the dense layer which follows the pooling layer downward.
//
static void backpropag__after_fully_connected(void * _Nonnull neural, unsigned int op) {
    
    extern tensor * propag_buffer;
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    int length = nn->conv2d->parameters->topology[op][1] * nn->conv2d->parameters->topology[op][2] *
                 nn->conv2d->parameters->topology[op][3];
    
    float buffer[length];
    memset(buffer, 0.0f, sizeof(buffer));
    
    cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->conv2d->dense_weights->shape[0][0][0], (int)nn->conv2d->dense_weights->shape[0][1][0], 1.0, nn->conv2d->dense_weights->val, (int)nn->conv2d->dense_weights->shape[0][1][0], propag_buffer->val, 1, 0.0, buffer, 1);
    memcpy(propag_buffer->val, buffer, length*sizeof(float));
}

//
// This routine is used to propagate the error from a convolution layer going upward into the network.
// It computes the term delta^{l+1}*rot{k^{l+1}}, where * is the convolution operation, delta^{l+1} and
// k{l+1} are the error (stored in the deltas_buffer array) and the weights of the convolution layer which
// follows the pooling layer downward.
//
static void backpropag_pooling_after_convolution(void * _Nonnull neural, unsigned int op,  int * _Nullable advance2) {
    
    transpose_convolution_op(neural, op, advance2);
}

// ------------------------------------------------------------------------------
// ---- This routine computes the delta_{l} at the current convolution layer and
// ---- updates the derivatives of the cost function with respect to the weights
// ---- and biases using the unrolled convolution.
// ---- nabla_W = trans(X) * nabla_Y (Chellapilla et al, 2006)
// ------------------------------------------------------------------------------
void backpropag_convolution_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance1, int * _Nullable advance2, int  * _Nullable advance3) {
    
    extern tensor *conv_input_matrix;
    extern tensor * propag_buffer;
    brain_storm_net *nn = (brain_storm_net *)neural;
    
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
        for (int i=0; i<nn->conv2d->conv_affine_transforms->rank; i++) {
            step = step * nn->conv2d->conv_affine_transforms->shape[l][i][0];
        }
        offset_z = offset_z + step;
    }
    
    // The term delta_{l}
    if (nn->conv2d->parameters->topology[op+1][0] == FULLY_CONNECTED) {
        backpropag__after_fully_connected(neural, op);
        int stride = 0;
        for (int k=0; k<q; k++) {
            for (int i=0; i<fh*fw; i++) {
                 nn->conv2d->deltas_buffer->val[stride+i] = propag_buffer->val[stride+i] * nn->conv2d->activation_derivatives[activ_idx](nn->conv2d->conv_affine_transforms->val[offset_z+(stride+i)]);
            }
            stride = stride + (fh * fw);
        }
        
    } else if (nn->conv2d->parameters->topology[op+1][0] == POOLING) {
        // We get the term delta^{l+1}*rot{k^{l+1}} or transpose(w^{l+1}) x delta^{l+1}
        // upsampled from a pooling layer
        int stride = 0;
        for (int k=0; k<q; k++) {
            for (int i=0; i<fh*fw; i++) {
                 nn->conv2d->deltas_buffer->val[stride+i] = nn->conv2d->deltas_buffer->val[stride+i] * nn->conv2d->activation_derivatives[activ_idx](nn->conv2d->conv_affine_transforms->val[offset_z+(stride+i)]);
            }
            stride = stride + (fh * fw);
        }
        
    } else if (nn->conv2d->parameters->topology[op+1][0] == CONVOLUTION) {
        // We get delta_{l+1} with a transpose convolution and compute here delta^{l+1}*rot{k^{l+1}}
        
        transpose_convolution_op(neural, op, advance2);
        
        int stride = 0;
        for (int k=0; k<p; k++) {
            for (int i=0; i<fh*fw; i++) {
                nn->conv2d->deltas_buffer->val[stride+i] = propag_buffer->val[stride+i] * nn->conv2d->activation_derivatives[activ_idx](nn->conv2d->conv_affine_transforms->val[offset_z+(stride+i)]);
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
        for (int i=0; i<nn->conv2d->conv_batch_cost_weight_derivs->rank; i++) {
            step = step * nn->conv2d->conv_batch_cost_weight_derivs->shape[l][i][0];
        }
        offset_w = offset_w + step;
    }
    
    // Offset to dC/db
    int offset_b = 0;
    for (int l=0; l<*advance2; l++) {
        int step = 1;
        for (int i=0; i<nn->conv2d->conv_batch_cost_bias_derivs->rank; i++) {
            step = step * nn->conv2d->conv_batch_cost_bias_derivs->shape[l][i][0];
        }
        offset_b = offset_b + step;
    }
    
    // Update dC/dw using convolutions (cross-correlations)
    // implemented with a matrix-matrix operation
    
    float C[p*(kh*kw)][q];
    float B[fh*fw][q];
    update_conv_input(neural, offset_a, p, fh, fw, fh_p, fw_p, kh, kw, sh, sw);
    
    int stride = 0;
    for (int k=0; k<q; k++) {
        int m = 0;
        for (int i=0; i<fh*fw; i++) {
            B[m][k] = propag_buffer->val[stride+i];
            m++;
        }
        stride = stride + (fh * fw);
    }
    
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, p*(kh*kw), q, fh*fw, 1.0f,  conv_input_matrix->val, p*(kh*kw), *B, q, 0.0f, *C, q);
    
    // Recover the dC/dw from the matrix-matrix operation
    int stride2 = 0;
    for (int k=0; k<q; k++) {
        int stride1 = 0;
        int indx = 0;
        for (int ll=0; ll<p; ll++) {
            for (int i=0; i<kh*kw; i++) {
                nn->conv2d->conv_batch_cost_weight_derivs->val[offset_w+(stride1+(stride2+i))] = C[indx][k];
                indx++;
            }
             stride1 = stride1 + (q * kh * kw);
        }
        stride2 = stride2 + (kh * kw);
    }
    
    // Update dC/db
    
    stride = 0;
    for (int l=0; l<q; l++) {
        float sum_b = 0.0f;
#ifdef __APPLE__
        vDSP_sve(nn->conv2d->deltas_buffer->val+stride, 1, &sum_b, fh*fw);
#else
        for (int i=0; i<fh*fw; i++) {
            sum_b = sum_b + nn->conv2d->deltas_buffer->val[stride+i];
        }
#endif
        nn->conv2d->conv_batch_cost_bias_derivs->val[offset_b+l] = sum_b;
        stride = stride + (fh * fw);
    }
    
    (*advance2)--;
    activ_idx--;
}

void backpropag_max_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance1, int * _Nullable advance2, int  * _Nullable advance3) {
    
    extern tensor * propag_buffer;
    static bool check = false;
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    if (!check) {
        if (*advance1 == 0) fatal(DEFAULT_CONSOLE_WRITER, "topology error in convolutional network. Using a pooling layer as an output layer is not permitted.");
        check = true;
    }
    
    int offset = 0;
    for (int l=0; l<*advance3; l++) {
        int step = 1;
        for (int i=0; i<nn->conv2d->max_pool_indexes->rank; i++) {
            step = step * nn->conv2d->max_pool_indexes->shape[l][i][0];
        }
        offset = offset + step;
    }
    
    unsigned int q = nn->conv2d->parameters->topology[op-1][1];
    unsigned int fh = nn->conv2d->parameters->topology[op][2];
    unsigned int fw = nn->conv2d->parameters->topology[op][3];
    
    if (nn->conv2d->parameters->topology[op+1][0] == CONVOLUTION) {
        backpropag_pooling_after_convolution((void *)nn, op, advance2);
    } else if (nn->conv2d->parameters->topology[op+1][0] == FULLY_CONNECTED) {
        backpropag__after_fully_connected((void *)nn, op);
    } else {
        unsigned int p = nn->conv2d->parameters->topology[op][1];
        memcpy(propag_buffer->val, nn->conv2d->deltas_buffer->val, (p*fh*fw)*sizeof(float));
    }
    
    // Upsampling: The error is directly assigned to the winning unit
    memset(nn->conv2d->deltas_buffer->val, 0.0f, nn->conv2d->deltas_buffer->shape[0][0][0]*sizeof(float));
    int stride = 0;
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh*fw; i++) {
            int idx = nn->conv2d->max_pool_indexes->int32_val[offset+(stride+i)];
            nn->conv2d->deltas_buffer->val[idx] = propag_buffer->val[stride+i];
        }
        stride = stride + (fh * fw);
    }
    
    (*advance3)--;
}

void backpropag_l2_pooling_op(void * _Nonnull neural, unsigned int op, int *_Nullable advance1, int * _Nullable advance2, int  * _Nullable advance3) {
    
}

void backpropag_average_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance1, int * _Nullable advance2, int  * _Nullable advance3) {
    
    extern tensor * propag_buffer;
    static bool check = false;
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    if (!check) {
        if (*advance1 == 0) fatal(DEFAULT_CONSOLE_WRITER, "topology error in convolutional network. Using a pooling layer as an output layer is not permitted.");
        check = true;
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
    
    if (nn->conv2d->parameters->topology[op+1][0] == CONVOLUTION) {
        backpropag_pooling_after_convolution((void *)nn, op, advance2);
    } else if (nn->conv2d->parameters->topology[op+1][0] == FULLY_CONNECTED) {
        backpropag__after_fully_connected((void *)nn, op);
    } else {
        unsigned int p = nn->conv2d->parameters->topology[op][1];
        memcpy(propag_buffer->val, nn->conv2d->deltas_buffer->val, (p*fh*fw)*sizeof(float));
    }
    
    int stride_c = 0;
    int stride_s = 0;
    for (int k=0; k<q; k++) {
        for (int i=0; i<fh; i++) {
            for (int j=0; j<fw; j++) {
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        nn->conv2d->deltas_buffer->val[stride_c+(((i*sh+u)*cols_c)+(j*sw+v))] =
                                1.0/((float)(kh*kw))*propag_buffer->val[stride_s+((i*fw)+j)];
                    }
                }
            }
        }
        stride_c = stride_c + (nn->conv2d->parameters->topology[op-1][2] * nn->conv2d->parameters->topology[op-1][3]);
        stride_s = stride_s + (fh * fw);
    }
}

void backpropag_in_conv2d_net(void * _Nonnull neural,
                              void (* _Nullable ptr_inference_func)(void * _Nonnull self)) {
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    // Activations at the input layer
    //TODO: channels > 1?
    int fh = nn->batch_inputs->shape[0][1][0];
    int fw = nn->batch_inputs->shape[0][2][0];
    int channels = nn->batch_inputs->shape[0][3][0];
    int stride1 = nn->example_idx * (fh * fw * channels);
    memcpy(nn->conv2d->conv_activations->val, nn->batch_inputs->val+stride1, (fh*fw*channels)*sizeof(float));
    
    // Inference (forward pass)
    ptr_inference_func(neural);
    
    // Backpropagation
    int advance1 = 0;
    int advance2 = nn->conv2d->num_conv2d_layers - 1;
    int advance3 = nn->conv2d->num_max_pooling_layers - 1;
    for (int l=nn->conv2d->num_backpropag_ops; l>=1; l--) {
        nn->conv2d->backpropag_ops[l-1](neural, l, &advance1, &advance2, &advance3);
        advance1++;
    }
}

void batch_accumulation_in_conv2d_net(void * _Nonnull neural) {
    
    // Accumulate dC/dw and dC/db at convolution and
    // fully connected layers
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    // Convolution layers
    int offset_w = 0;
    int offset_b = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned int p = nn->conv2d->conv_batch_cost_weight_derivs->shape[l][0][0];
        unsigned int q = nn->conv2d->conv_batch_cost_weight_derivs->shape[l][1][0];
        unsigned int kh = nn->conv2d->conv_batch_cost_weight_derivs->shape[l][2][0];
        unsigned int kw = nn->conv2d->conv_batch_cost_weight_derivs->shape[l][3][0];
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                for (int u=0; u<kh; u++) {
                    for (int v=0; v<kw; v++) {
                        nn->conv2d->conv_cost_weight_derivs->val[offset_w+(stride1+(stride2+(u*kw+v)))] =
                          nn->conv2d->conv_cost_weight_derivs->val[offset_w+(stride1+(stride2+(u*kw+v)))] +
                              nn->conv2d->conv_batch_cost_weight_derivs->val[offset_w+(stride1+(stride2+(u*kw+v)))];
                    }
                }
                stride2 = stride2 + (kh * kw);
            }
            stride1 = stride1 + (q * kh * kw);
        }
        
        for (int ll=0; ll<q; ll++) {
            nn->conv2d->conv_cost_bias_derivs->val[offset_b+ll] = nn->conv2d->conv_cost_bias_derivs->val[offset_b+ll] + nn->conv2d->conv_batch_cost_bias_derivs->val[offset_b+ll];
        }
        
        offset_w = offset_w + (p * q * kh * kw);
        offset_b = offset_b + q;
    }
    
    // Fully connected layers
    offset_w = 0;
    offset_b = 0;
    for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
        unsigned int m = nn->conv2d->dense_cost_weight_derivs->shape[l][0][0];
        unsigned int n = nn->conv2d->dense_cost_weight_derivs->shape[l][1][0];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->conv2d->dense_cost_weight_derivs->val[offset_w+((i*n)+j)] =
                   nn->conv2d->dense_cost_weight_derivs->val[offset_w+((i*n)+j)] + nn->conv2d->dense_batch_cost_weight_derivs->val[offset_w+((i*n)+j)];
            }
        }
        for (int i=0; i<m; i++) {
            nn->conv2d->dense_cost_bias_derivs->val[offset_b+i] =
            nn->conv2d->dense_cost_bias_derivs->val[offset_b+i] + nn->conv2d->dense_batch_cost_bias_derivs->val[offset_b+i];
        }
        
        offset_w = offset_w + (m * n);
        offset_b = offset_b + m;
    }
}
