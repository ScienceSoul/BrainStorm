//
//  main.c
//  BrainStormTest
//
//  Created by Hakime Seddik on 18/09/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#include <Accelerate/Accelerate.h>

#include "BrainStorm.h"
#include "Conv2DNetOps.h"

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

// The intial 8 * 8 feeding activations used to test the pooling routines
// and the resulting outputs after a 2 * 2 pooling kernel is applied two
// times. Once to the input and the second time to the result of the first
// pooling.
// At the first pooling layer, the maps size is 7 * 7.
// At the second polling layer, the maps size is 6 * 6.

// Input:
//  | 1  2  3  4  5  6  7  8  |
//  | 9  10 11 12 13 14 15 16 |
//  | 17 18 19 20 21 22 23 24 |
//  | 25 26 27 28 29 30 31 32 |
//  | 33 34 35 36 37 38 39 40 |
//  | 41 42 43 44 45 46 47 48 |
//  | 49 50 51 52 53 54 55 56 |
//  | 57 58 59 60 61 62 63 64 |
float mat_feed[8*8] = {1,  2,  3,  4,  5,  6,  7,  8,
                            9,  10, 11, 12, 13, 14, 15, 16,
                            17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28, 29, 30, 31, 32,
                            33, 34, 35, 36, 37, 38, 39, 40,
                            41, 42, 43, 44, 45, 46, 47, 48,
                            49, 50, 51, 52, 53, 54, 55, 56,
                            57, 58, 59, 60, 61, 62, 63, 64};

// Unit strides max pooling
//  | 10 11 12 13 14 15 16 |
//  | 18 19 20 21 22 23 24 |
//  | 26 27 28 29 30 31 32 |
//  | 34 35 36 37 38 39 40 |
//  | 42 43 44 45 46 47 48 |
//  | 50 51 52 53 54 55 56 |
//  | 58 59 60 61 62 63 64 |
//
//  | 19 20 21 22 23 24 |
//  | 27 28 29 30 31 32 |
//  | 35 36 37 38 39 40 |
//  | 43 44 45 46 47 48 |
//  | 51 52 53 54 55 56 |
//  | 59 60 61 62 63 64 |
float mat_max_pool1[7*7] = {10, 11, 12, 13, 14, 15, 16,
                            18, 19, 20, 21, 22, 23, 24,
                            26, 27, 28, 29, 30, 31, 32,
                            34, 35, 36, 37, 38, 39, 40,
                            42, 43, 44, 45, 46, 47, 48,
                            50, 51, 52, 53, 54, 55, 56,
                            58, 59, 60, 61, 62, 63, 64};

float mat_max_pool2[6*6] = {19, 20, 21, 22, 23, 24,
                            27, 28, 29, 30, 31, 32,
                            35, 36, 37, 38, 39, 40,
                            43, 44, 45, 46, 47, 48,
                            51, 52, 53, 54, 55, 56,
                            59, 60, 61, 62, 63, 64};

// Non-unit strides max pooling (strides=2)
//  | 10 12 14 16 |
//  | 26 28 30 32 |
//  | 42 44 46 48 |
//  | 58 60 62 64 |
//
//  | 28 32 |
//  | 60 64 |
float mat_max_pool_stride_1[4*4] = {10, 12, 14, 16,
                                    26, 28, 30, 32,
                                    42, 44, 46, 48,
                                    58, 60, 62, 64};

float mat_max_pool_stride_2[2*2] = {28, 32,
                                    60, 64};

// Unit strides average pooling
//  | 5.5  6.5  7.5  8.5  9.5  10.5 11.5 |
//  | 13.5 14.5 15.5 16.5 17.5 18.5 19.5 |
//  | 21.5 22.5 23.5 24.5 25.5 26.5 27.5 |
//  | 29.5 30.5 31.5 32.5 33.5 34.5 35.5 |
//  | 37.5 38.5 39.5 40.5 41.5 42.5 43.5 |
//  | 45.5 46.5 47.5 48.5 49.5 50.5 51.5 |
//  | 53.5 54.5 55.5 56.5 57.5 58.5 59.5 |
//
//  | 10 11 12 13 14 15 |
//  | 18 19 20 21 22 23 |
//  | 26 27 28 29 30 31 |
//  | 34 35 36 37 38 39 |
//  | 42 43 44 45 46 47 |
//  | 50 51 52 53 54 55 |
float mat_average_pool1[7*7] = {5.5,  6.5,  7.5,  8.5,  9.5,  10.5, 11.5,
                                13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5,
                                21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5,
                                29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5,
                                37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5,
                                45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5,
                                53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5};

float mat_average_pool2[6*6] = {10, 11, 12, 13, 14, 15,
                                18, 19, 20, 21, 22, 23,
                                26, 27, 28, 29, 30, 31,
                                34, 35, 36, 37, 38, 39,
                                42, 43, 44, 45, 46, 47,
                                50, 51, 52, 53, 54, 55};

// Non-unit strides average pooling (strides=2)
//  | 5.5  7.5  9.5  11.5 |
//  | 21.5 23.5 25.5 27.5 |
//  | 37.5 39.5 41.5 43.5 |
//  | 53.5 55.5 57.5 59.5 |
//
//  | 14.5 18.5 |
//  | 46.5 50.5 |
float mat_average_pool_stride_1[4*4] = {5.5,  7.5,  9.5,  11.5,
                                        21.5, 23.5, 25.5, 27.5,
                                        37.5, 39.5, 41.5, 43.5,
                                        53.5, 55.5, 57.5, 59.5};

float mat_average_pool_stride_2[2*2] = {14.5, 18.5,
                                        46.5, 50.5};

// Unit strides L2 pooling
//  | 13.63  15.29  17.02  18.81  20.63  22.49  24.37  |
//  | 28.17  30.09  32.03  33.97  35.91  37.86  39.82  |
//  | 43.74  45.71  47.68  49.65  51.63  53.60  55.58  |
//  | 59.54  61.53  63.51  65.49  67.48  69.46  71.45  |
//  | 75.43  77.42  79.41  81.40  83.39  85.38  87.37  |
//  | 91.35  93.34  95.34  97.33  99.32  101.32 103.31 |
//  | 107.30 109.29 111.29 113.28 115.28 117.27 119.27 |
//
//  | 46.04  49.55  53.14  56.78  60.46  64.18  |
//  | 75.52  79.34  83.18  87.04  90.90  94.78  |
//  | 106.47 110.38 114.29 118.22 122.14 126.07 |
//  | 137.89 141.84 145.79 149.74 153.70 157.65 |
//  | 169.54 173.50 177.47 181.43 185.40 189.37 |
//  | 201.29 205.27 209.24 213.22 217.20 221.17 |
float mat_l2_pool1[7*7] = {13.63,  15.29,  17.02,  18.81,  20.63,  22.49,  24.37,
                           28.17,  30.09,  32.03,  33.97,  35.91,  37.86,  39.82,
                           43.74,  45.71,  47.68,  49.65,  51.63,  53.60,  55.58,
                           59.54,  61.53,  63.51,  65.49,  67.48,  69.46,  71.45,
                           75.43,  77.42,  79.41,  81.40,  83.39,  85.38,  87.37,
                           91.35,  93.34,  95.34,  97.33,  99.32,  101.32, 103.31,
                           107.30, 109.29, 111.29, 113.28, 115.28, 117.27, 119.27};

float mat_l2_pool2[6*6] = {46.04, 49.55, 53.14, 56.78, 60.46, 64.18,
                           75.52, 79.34, 83.18, 87.04, 90.90, 94.78,
                           106.47, 110.38, 114.29, 118.22, 122.14, 126.07,
                           137.89, 141.84, 145.79, 149.74, 153.70, 157.65,
                           169.54, 173.50, 177.47, 181.43, 185.40, 189.37,
                           201.29, 205.27, 209.24, 213.22, 217.20, 221.17};

// Non-unit strides L2 pooling (strides=2)
//  | 13.63  17.02  20.63  24.37   |
//  | 43.74  47.68  51.63  55.58   |
//  | 75.43  79.41  83.39  87.37   |
//  | 107.30 111.29 115.28 119.27  |
//
//  | 68.29  82.31  |
//  | 189.46 205.19 |
float mat_l2_pool_stride_1[4*4] = { 13.63,  17.02,  20.63,  24.37,
                                    43.74,  47.68,  51.63,  55.58,
                                    75.43,  79.41,  83.39,  87.37,
                                    107.30, 111.29, 115.28, 119.27};

float mat_l2_pool_stride_2[2*2] = {68.29, 82.31,
                                   189.46, 205.19};

// The following is used for the tests involving convolution
// and pooling
float kernel[5*5] = {0.1, 0.2, 0.3, 0.4, 0.5,
                     0.2, 0.3, 0.4, 0.5, 0.6,
                     0.3, 0.4, 0.5, 0.6, 0.7,
                     0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9, 1.0, 1.1, 1.2, 1.3};
// The reference pooling from the convolution
//  | 329 343.5 358 |
//  | 445 459.5 474 |
//  | 561 575.5 590 |
float ref_pool[3*3] = {329, 343.5, 358,
                       445, 459.5, 474,
                       561, 575.5, 590};

// The array of pointers to pooling results
float *ptr[2];

// Formating functions
typedef float (*func_ptr)(float val);

float non_truncated(float val) {
    return val;
}
float truncated(float val) {
    return floorf(val*100.0f)/100.f;
}

float activation_func(float val, float * _Nullable dummy1, unsigned int * _Nullable dummy2) {
    return val;
}

void init_feed_activations(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Initialize the feeding layer with the input:
    for (int l=0; l<1; l++) {
        
        unsigned int p  = nn->conv2d->conv_activations->shape[l][0][0];
        unsigned int fh = nn->conv2d->conv_activations->shape[l][1][0];
        unsigned int fw = nn->conv2d->conv_activations->shape[l][2][0];
        
        int stride = 0;
        for (int k=0; k<p; k++) {
            int idx = 0;
            for (int i=0; i<fh; i++) {
                for (int j=0; j<fw; j++) {
                    nn->conv2d->conv_activations->val[stride+(i*fw+j)] = mat_feed[idx];
                    idx++;
                }
            }
            stride = stride + (fh * fw);
        }
    }
}

void set_up(void * _Nonnull neural, int * _Nonnull maps_size, unsigned int number_of_maps, unsigned int kh, unsigned int sh, enum convlution_layer_types layer) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l == 0) {
            nn->conv2d->parameters->topology[l][0] = FEED;
            nn->conv2d->parameters->topology[l][1] = number_of_maps;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
        } else {
            nn->conv2d->parameters->topology[l][0] = layer;
            nn->conv2d->parameters->topology[l][1] = number_of_maps;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
            nn->conv2d->parameters->topology[l][4] = kh;
            nn->conv2d->parameters->topology[l][5] = kh;
            nn->conv2d->parameters->topology[l][6] = sh;
            nn->conv2d->parameters->topology[l][7] = sh;
        }
    }
    
    // The activation tensors:
    // Pooling tests:
    // a1 = shape[number_of_maps,maps_size[l],maps_size[l]] = shape[6,8,8]
    // a2 = shape[number_of_maps,maps_size[l],maps_size[l]] = shape[6,7,7]
    // a3 = shape[number_of_maps,maps_size[l],maps_size[l]] = shape[6,6,6]
    //
    // Convolution test:
    // a1 = shape[1,8,8]
    // a2 = shape[1,4,4]
    tensor_dict *dict = init_tensor_dict();
    dict->rank = 3;
    nn->conv2d->conv_activations = (tensor *)nn->conv2d->conv_activations_alloc(neural, (void *)dict, true);
    free(dict);
    
    init_feed_activations(neural);
}

void init_convol_kernels(void * _Nonnull neural, unsigned int kh) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    int offset = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned p = nn->conv2d->conv_weights->shape[l][0][0];
        unsigned q = nn->conv2d->conv_weights->shape[l][1][0];
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                int idx = 0;
                for (int i=0; i<kh; i++) {
                    for (int j=0; j<kh; j++) {
                        nn->conv2d->conv_weights->val[offset+(stride1+(stride2+((i*kh)+j)))] = kernel[idx];
                        idx++;
                    }
                }
                stride2 = stride2 + (kh * kh);
            }
            stride1 = stride1 + (q * kh * kh);
        }
        offset = offset + (p * q * kh * kh);
    }
}

void init_dense_weights(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    // Init dense weights with ones
    int offset = 0;
    for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
        
        unsigned int m = nn->conv2d->dense_weights->shape[l][0][0];
        unsigned int n = nn->conv2d->dense_weights->shape[l][1][0];
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->conv2d->dense_weights->val[offset+((i*n)+j)] = 1.0f;
            }
        }
        offset = offset + (m * n);
    }
}

bool check_activations(void * _Nonnull neural, func_ptr formater) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    int offset = 0;
    int p_idx  = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        
        unsigned int p = nn->conv2d->conv_activations->shape[l][0][0];
        unsigned int fh = nn->conv2d->conv_activations->shape[l][1][0];
        unsigned int fw = nn->conv2d->conv_activations->shape[l][2][0];
        
        if (l > 0) {
            int stride = 0;
            for (int k=0; k<p; k++) {
                int idx = 0;
                for (int i=0; i<fh; i++) {
                    for (int j=0; j<fw; j++) {
                        if (formater(nn->conv2d->conv_activations->val[offset+(stride+((i*fw)+j))]) != *(ptr[p_idx]+idx)) {
                            return false;
                        }
                        idx++;
                    }
                }
                stride = stride + (fh * fw);
            }
            p_idx++;
        }
        offset = offset + (p * fh * fw);
    }
    return true;
}


void ref_convol(float * _Nonnull ref_conv, unsigned int kh) {
    
    float C[8*8];
    float flipped_kernel[5*5];
    float flip[5*5] = {0, 0, 0, 0, 1,
                       0, 0, 0, 1, 0,
                       0, 0, 1, 0 ,0,
                       0, 1, 0, 0, 0,
                       1, 0, 0, 0 ,0};
    flip_mat(kh, kh, kh, flip, kh, kernel, kh, flipped_kernel, kh);
    vDSP_f5x5(mat_feed, 8, 8, flipped_kernel, C);
    
    int idx = 0;
    int k = 0;
    for (int i=0; i<8; i++) {
        for (int j=0; j<8; j++) {
            if (C[idx] != 0.0f) {
                ref_conv[k] = truncated(C[idx]);
                k++;
            }
            idx++;
        }
    }
}
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

bool test_create_tensors(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int kh = 5;
    unsigned int layers = 10;
    float init_value = 1.0f;
    int vec[5];
    
    tensor_dict *dict = init_tensor_dict();
    dict->flattening_length = layers;
    dict->init_with_value = true;
    dict->init_value = init_value;
    
    dict->rank = 1;
    vec[0] = kh;
    shape(dict->shape, layers, dict->rank, vec);
    tensor *tensor_1d = (tensor *)nn->tensor(neural, *dict);
    if (tensor_1d == NULL) {
        return false;
    }
    int offset = 0;
    for (int l=0; l<layers; l++) {
        for (int i=0; i<kh; i++) {
            if (tensor_1d->val[offset+i] != init_value) {
                return false;
            }
        }
        offset = offset + kh;
    }
    free(tensor_1d->val);
    free(tensor_1d);
    fprintf(stdout, ">>>>>>> Test: tensor 1D: success...\n");
    
    dict->rank = 2;
    vec[1] = kh;
    shape(dict->shape, layers, dict->rank, vec);
    tensor *tensor_2d = (tensor *)nn->tensor(neural, *dict);
    if (tensor_2d == NULL) {
        return false;
    }
    offset = 0;
    for (int l=0; l<layers; l++) {
        for (int i=0; i<kh; i++) {
            for (int j=0; j<kh; j++) {
                if (tensor_2d->val[offset+((i*kh)+j)] != init_value) {
                    return false;
                }
            }
        }
        offset = offset + (kh * kh);
    }
    free(tensor_2d->val);
    free(tensor_2d);
    fprintf(stdout, ">>>>>>> Test: tensor 2D: success...\n");
    
    dict->rank = 3;
    vec[2] = kh;
    shape(dict->shape, layers, dict->rank, vec);
    tensor *tensor_3d = (tensor *)nn->tensor(neural, *dict);
    if (tensor_3d == NULL) {
        return false;;
    }
    offset = 0;
    for (int l=0; l<layers; l++) {
        int stride = 0;
        for (int k=0; k<kh; k++) {
            for (int i=0; i<kh; i++) {
                for (int j=0; j<kh; j++) {
                    if (tensor_3d->val[offset+(stride+((i*kh)+j))] != init_value) {
                        return false;
                    }
                }
            }
            stride = stride + (kh * kh);
        }
        offset = offset + (kh * kh * kh);
    }
    fprintf(stdout, ">>>>>>> Test: tensor 3D: success...\n");
    free(tensor_3d->val);
    free(tensor_3d);
    
    dict->rank = 4;
    vec[3] = kh;
    shape(dict->shape, layers, dict->rank, vec);
    tensor *tensor_4d = (tensor *)nn->tensor(neural, *dict);
    if (tensor_4d == NULL) {
        return false;
    }
    offset = 0;
    for (int l=0; l<layers; l++) {
        int stride1 = 0;
        for (int k=0; k<kh; k++) {
            int stride2 = 0;
            for (int ll=0; ll<kh; ll++) {
                for (int i=0; i<kh; i++) {
                    for (int j=0; j<kh; j++) {
                        if (tensor_4d->val[offset+(stride1+(stride2+((i*kh)+j)))] != init_value) {
                            return false;
                        }
                    }
                }
                stride2 = stride2 + (kh * kh);
            }
            stride1 = stride1 + (kh * kh * kh);
        }
        offset = offset + (kh * kh * kh * kh);
    }
    fprintf(stdout, ">>>>>>> Test: tensor 4D: success...\n");
    free(tensor_4d->val);
    free(tensor_4d);
    
    dict->rank = 5;
    vec[4] = kh;
    shape(dict->shape, layers, dict->rank, vec);
    tensor *tensor_5d = (tensor *)nn->tensor(neural, *dict);
    if (tensor_5d == NULL) {
        return false;
    }
    offset = 0;
    for (int l=0; l<layers; l++) {
        int stride1 = 0;
        for (int k=0; k<kh; k++) {
            int stride2 = 0;
            for (int ll=0; ll<kh; ll++) {
                int stride3 = 0;
                for (int ll=0; ll<kh; ll++) {
                    for (int i=0; i<kh; i++) {
                        for (int j=0; j<kh; j++) {
                            if (tensor_5d->val[offset+(stride1+(stride2+(stride3+((i*kh)+j))))] != init_value) {
                                return false;
                            }
                        }
                    }
                    stride3 = stride3 + (kh * kh);
                }
                stride2 = stride2 + (kh * kh * kh);
            }
            stride1 = stride1 + (kh * kh * kh * kh);
        }
        offset = offset + (kh * kh * kh * kh * kh);
    }
    fprintf(stdout, ">>>>>>> Test: tensor 5D: success...\n");
    free(tensor_5d->val);
    free(tensor_5d);
    
    free(dict);
    
    return true;
}

bool test_kernels_flipping(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int kh = 5;
    
    // The weights tensors
    // t1 = shape[1,6,5,5]
    // t2 = shape[6,12,5,5]
    nn->conv2d->num_conv2d_layers = 2;
    tensor_dict *dict = init_tensor_dict();
    dict->rank = 4;
    int maps[3] = {1, 6, 12};
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4] = {maps[l], maps[l+1], kh, kh};
        shape(dict->shape, dict->rank, vec, l);
    }
    dict->flattening_length = nn->conv2d->num_conv2d_layers;
    nn->conv2d->conv_weights = (tensor *)nn->tensor(neural, *dict);
    nn->conv2d->flipped_weights = (tensor *)nn->tensor(neural, *dict);
    
    // The flipping matrix is of the form:
    // | 0  0  0  0  1  |
    // | 0  0  0  1  0  |
    // | 0  0  1  0  0  |
    // | 0  1  0  0  0  |
    // | 1  0  0  0  0  |
    float flip[5][5] = {{0, 0, 0, 0, 1},
                        {0, 0, 0, 1, 0},
                        {0, 0, 1, 0 ,0},
                        {0, 1, 0, 0, 0},
                        {1, 0, 0, 0 ,0}};
    dict->rank = 2;
    int vec[2] = {kh, kh};
    shape(dict->shape, nn->conv2d->num_conv2d_layers, dict->rank, vec);
    nn->conv2d->flip_matrices = (tensor *)nn->tensor(neural, *dict);
    nn->create_flip(neural);
    
    int offset = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        for (int i=0; i<kh; i++) {
            for (int j=0; j<kh; j++) {
                if (nn->conv2d->flip_matrices->val[offset+((i*kh)+j)] != flip[i][j]) {
                    fprintf(stdout, "error: wrong flip matrix.\n");
                    return false;
                }
            }
        }
        offset = offset + (kh * kh);
    }
    
    // Initialize the kernel (weight) matrices with:
    // | 1  2  3  4  5  |
    // | 6  7  8  9  10 |
    // | 11 12 13 14 15 |
    // | 16 17 18 19 20 |
    // | 21 22 23 24 25 |
    offset = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned p = maps[l];
        unsigned q = maps[l+1];
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                float value = 1.0f;
                for (int i=0; i<kh; i++) {
                    for (int j=0; j<kh; j++) {
                        nn->conv2d->conv_weights->val[offset+(stride1+(stride2+((i*kh)+j)))] = value;
                        value = value + 1.0f;
                    }
                }
                stride2 = stride2 + (kh * kh);
            }
            stride1 = stride1 + (q * kh * kh);
        }
        offset = offset + (p * q * kh * kh);
    }
    
    // Flip the kernels (weights)
    nn->flip_kernels(neural);
    
    // Check the result, flipped kernels should be:
    // | 25 24 23 22 21 |
    // | 20 19 18 17 16 |
    // | 15 14 13 12 11 |
    // | 10  9  8  7  6 |
    // |  5  4  3  2  1 |
    float mat_flipped[5][5] = {{25, 24, 23, 22, 21},
                               {20, 19, 18, 17, 16},
                               {15, 14, 13, 12, 11},
                               {10,  9,  8,  7,  6},
                               { 5,  4,   3, 2,  1}};
    offset = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned p = nn->conv2d->flipped_weights->shape[l][0][0];
        unsigned q = nn->conv2d->flipped_weights->shape[l][1][0];
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                for (int i=0; i<kh; i++) {
                    for (int j=0; j<kh; j++) {
                        if (nn->conv2d->flipped_weights->val[offset+(stride1+(stride2+((i*kh)+j)))] != mat_flipped[i][j]) {
                            return false;
                        }
                    }
                }
                stride2 = stride2 + (kh * kh);
            }
            stride1 = stride1 + (q * kh * kh);
        }
        offset = offset + (p * q * kh * kh);
    }
    
    free(nn->conv2d->conv_weights->val);
    free(nn->conv2d->conv_weights);
    
    free(nn->conv2d->flipped_weights->val);
    free(nn->conv2d->flipped_weights);
    
    free(nn->conv2d->flip_matrices->val);
    free(nn->conv2d->flip_matrices);
    
    free(dict);
    
    fprintf(stdout, ">>>>>>> Test: tensor flipping: success...\n");
    return true;
}

bool test_create_convol_matrix(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    //---------------------------------------
    // Test with unit strides
    //---------------------------------------
    
    unsigned int kh = 2;
    unsigned int sh = 1;
    nn->network_num_layers = 3;
    nn->conv2d->num_conv2d_layers = 2;
    
    // The weights tensors
    // t1 = shape[1,6,2,2]
    // t2 = shape[6,12,2,2]
    tensor_dict *dict = init_tensor_dict();
    dict->rank = 4;
    int maps[4] = {1, 6, 12};
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4] = {maps[l], maps[l+1], kh, kh};
        shape(dict->shape, dict->rank, vec, l);
    }
    dict->flattening_length = nn->conv2d->num_conv2d_layers;
    nn->conv2d->conv_weights = (tensor *)nn->tensor(neural, *dict);
    nn->conv2d->flipped_weights = (tensor *)nn->tensor(neural, *dict);
    
    dict->rank = 2;
    int vec[2] = {kh, kh};
    shape(dict->shape, nn->conv2d->num_conv2d_layers, dict->rank, vec);
    nn->conv2d->flip_matrices = (tensor *)nn->tensor(neural, *dict);
    nn->create_flip(neural);
    
    // The convolution matrix tensors
    // Assume 4 x 4 input layer. With 2 x 2 kernels and unit strides,
    // after the first convolution we get maps of size 3 x 3. After the
    // second convolution we get maps of size 2 x 2
    // The convolution matrix tensors are:
    //   c1 = shape[1,6,9,16]
    //   c2 = shape[6,12,4,9]
    dict->rank = 4;
    int maps_size[3] = {4, 3, 2};
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4] = {maps[l], maps[l+1],
                      maps_size[l+1] * maps_size[l+1],
                      maps_size[l] * maps_size[l]};
        shape(dict->shape, dict->rank, vec, l);
    }
    nn->conv2d->conv_matrices = (tensor *)nn->tensor(neural, *dict);
    
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l == 0) {
            nn->conv2d->parameters->topology[l][0] = FEED;
        }
        nn->conv2d->parameters->topology[l][3] = maps_size[l];
    }
    for (int l=1; l<nn->network_num_layers; l++) {
        nn->conv2d->parameters->topology[l][0] = CONVOLUTION;
        nn->conv2d->parameters->topology[l][6] = sh;
        nn->conv2d->parameters->topology[l][7] = sh;
    }
    
    // Initialize the kernel (weight) matrices with:
    // | 1  2 |
    // | 3  4 |
    int offset = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned p = maps[l];
        unsigned q = maps[l+1];
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                float value = 1.0f;
                for (int i=0; i<kh; i++) {
                    for (int j=0; j<kh; j++) {
                        nn->conv2d->conv_weights->val[offset+(stride1+(stride2+((i*kh)+j)))] = value;
                        value = value + 1.0f;
                    }
                }
                stride2 = stride2 + (kh * kh);
            }
            stride1 = stride1 + (q * kh * kh);
        }
        offset = offset + (p * q * kh * kh);
    }
    
    // Flip the kernels (weights) to get
    // | 4  3 |
    // | 2  1 |
    nn->flip_kernels(neural);
    
    // Create the convolution matrices
    nn->conv_mat_update(neural);
    
    // Check the results, the convolution matrices should be
    // First convolution matrix:
    //   | 4 3 0 0 2 1 0 0 0 0 0 0 0 0 0 0 |
    //   | 0 4 3 0 0 2 1 0 0 0 0 0 0 0 0 0 |
    //   | 0 0 4 3 0 0 2 1 0 0 0 0 0 0 0 0 |
    //   | 0 0 0 0 4 3 0 0 2 1 0 0 0 0 0 0 |
    //   | 0 0 0 0 0 4 3 0 0 2 1 0 0 0 0 0 |
    //   | 0 0 0 0 0 0 4 3 0 0 2 1 0 0 0 0 |
    //   | 0 0 0 0 0 0 0 0 4 3 0 0 2 1 0 0 |
    //   | 0 0 0 0 0 0 0 0 0 4 3 0 0 2 1 0 |
    //   | 0 0 0 0 0 0 0 0 0 0 4 3 0 0 2 1 |
    //
    // Second convolution matrix:
    //   | 4 3 0 2 1 0 0 0 0 |
    //   | 0 4 3 0 2 1 0 0 0 |
    //   | 0 0 0 4 3 0 2 1 0 |
    //   | 0 0 0 0 4 3 0 2 1 |
    float conv_mat1[9*16] = {4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1};
    
    float conv_mat2[4*9] = {4, 3, 0, 2, 1, 0, 0, 0, 0,
                            0, 4, 3, 0, 2, 1, 0, 0, 0,
                            0, 0, 0, 4, 3, 0, 2, 1, 0,
                            0, 0, 0, 0, 4, 3, 0, 2, 1};
    offset = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned int p = nn->conv2d->conv_matrices->shape[l][0][0];
        unsigned int q = nn->conv2d->conv_matrices->shape[l][1][0];
        unsigned int kh = nn->conv2d->conv_matrices->shape[l][2][0];
        unsigned int kw = nn->conv2d->conv_matrices->shape[l][3][0];
        
        float *ptr = NULL;
        if (l==0) {
            ptr = conv_mat1;
        } else ptr = conv_mat2;
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                int idx = 0;
                for (int i=0; i<kh; i++) {
                    for (int j=0; j<kw; j++) {
                        if (nn->conv2d->conv_matrices->val[offset+(stride1+(stride2+((i*kw)+j)))] != ptr[idx]) {
                            return false;
                        }
                        idx++;
                    }
                }
                stride2 = stride2 + (kh * kw);
            }
            stride1 = stride1 + (q * kh * kw);
        }
        offset = offset + (p * q * kh * kw);
    }
    
    fprintf(stdout, ">>>>>>> Test: convolution matrix (unit strides): success...\n");
    
    //---------------------------------------
    // Test with non-unit strides
    //---------------------------------------
    
    sh = 2;
    nn->network_num_layers = 2;
    nn->conv2d->num_conv2d_layers = 1;
    
    free(nn->conv2d->conv_matrices->val);
    free(nn->conv2d->conv_matrices);
    
    // Assume 4 x 4 input layer. With 2 x 2 kernel and strides of 2,
    // after the convolution we get maps of size 2 x 2.
    dict->rank = 4;
    int map_size_stride[2] = {4, 2};
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4] = {maps[l], maps[l+1],
                      map_size_stride[l+1] * map_size_stride[l+1],
                      map_size_stride[l] * map_size_stride[l]};
        shape(dict->shape, dict->rank, vec, l);
    }
    nn->conv2d->conv_matrices = (tensor *)nn->tensor(neural, *dict);
    
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l == 0) {
            nn->conv2d->parameters->topology[l][0] = FEED;
        }
        nn->conv2d->parameters->topology[l][3] = map_size_stride[l];
    }
    for (int l=1; l<nn->network_num_layers; l++) {
        nn->conv2d->parameters->topology[l][0] = CONVOLUTION;
        nn->conv2d->parameters->topology[l][6] = sh;
        nn->conv2d->parameters->topology[l][7] = sh;
    }
    
    nn->conv_mat_update(neural);
    
    // Check the results, the convolution matrix should be:
    //   | 4 3 0 0 2 1 0 0 0 0 0 0 0 0 0 0 |
    //   | 0 0 4 3 0 0 2 1 0 0 0 0 0 0 0 0 |
    //   | 0 0 0 0 0 0 0 0 4 3 0 0 2 1 0 0 |
    //   | 0 0 0 0 0 0 0 0 0 0 4 3 0 0 2 1 |
    float conv_mat_stride[4*16] = {4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1};
    offset = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned int p = nn->conv2d->conv_matrices->shape[l][0][0];
        unsigned int q = nn->conv2d->conv_matrices->shape[l][1][0];
        unsigned int kh = nn->conv2d->conv_matrices->shape[l][2][0];
        unsigned int kw = nn->conv2d->conv_matrices->shape[l][3][0];
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                int idx = 0;
                for (int i=0; i<kh; i++) {
                    for (int j=0; j<kw; j++) {
                        if (nn->conv2d->conv_matrices->val[offset+(stride1+(stride2+((i*kw)+j)))] != conv_mat_stride[idx]) {
                            return false;
                        }
                        idx++;
                    }
                }
                stride2 = stride2 + (kh * kw);
            }
            stride1 = stride1 + (q * kh * kw);
        }
        offset = offset + (p * q * kh * kw);
    }
    
    fprintf(stdout, ">>>>>>> Test: convolution matrix (non-unit strides): success...\n");
    
    free(nn->conv2d->conv_weights->val);
    free(nn->conv2d->conv_weights);
    
    free(nn->conv2d->flipped_weights->val);
    free(nn->conv2d->flipped_weights);
    
    free(nn->conv2d->flip_matrices->val);
    free(nn->conv2d->flip_matrices);
    
    free(nn->conv2d->conv_matrices->val);
    free(nn->conv2d->conv_matrices);
    
    free(dict);
    
    return true;
}

bool test_max_pooling(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    //---------------------------------------
    // Test with unit strides
    //---------------------------------------
    
    unsigned int kh = 2;
    unsigned int sh = 1;
    
    // Assume a network with three layers: one feeding layer
    // and two max pooling layers
    nn->network_num_layers = 3;
    nn->conv2d->num_pooling_layers = 2;
    nn->conv2d->num_infer_ops = 2;
    int idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l > 0) {
            nn->conv2d->inferenceOps[idx] = max_pooling_op;
            idx++;
        }
    }
    
    int maps_size[3] = {8, 7, 6};
    set_up(neural, maps_size, 6, kh, sh, POOLING);
    
    // Apply the pooling
    inference_in_conv2d_net(neural);
    
    // Check the result
    ptr[0] = mat_max_pool1;
    ptr[1] = mat_max_pool2;
    func_ptr formater = non_truncated;
    if (!check_activations(neural, formater)) {
        return false;
    }
    fprintf(stdout, ">>>>>>> Test: max pooling (unit strides): success...\n");
    
    //---------------------------------------
    // Test with non-unit strides
    //---------------------------------------
    
    free(nn->conv2d->conv_activations->val);
    free(nn->conv2d->conv_activations);
    
    sh = 2;
    maps_size[1] = 4;
    maps_size[2] = 2;
    set_up(neural, maps_size, 6, kh, sh, POOLING);
    
    // Apply the pooling
    inference_in_conv2d_net(neural);
    
     // Check the result
    ptr[0] = mat_max_pool_stride_1;
    ptr[1] = mat_max_pool_stride_2;
    if (!check_activations(neural, formater)) {
        return false;
    }
    fprintf(stdout, ">>>>>>> Test: max pooling (non-unit strides): success...\n");
    
    free(nn->conv2d->conv_activations->val);
    free(nn->conv2d->conv_activations);
    
    return true;
}

bool test_average_pooling(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    //---------------------------------------
    // Test with unit strides
    //---------------------------------------
    
    unsigned int kh = 2;
    unsigned int sh = 1;
    
    // Assume a network with three layers: one feeding layer
    // and two max pooling layers
    nn->network_num_layers = 3;
    nn->conv2d->num_pooling_layers = 2;
    nn->conv2d->num_infer_ops = 2;
    int idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l > 0) {
            nn->conv2d->inferenceOps[idx] = average_pooling_op;
            idx++;
        }
    }
    
    int maps_size[3] = {8, 7, 6};
    set_up(neural, maps_size, 6, kh, sh, POOLING);
    
    // Apply the pooling
    inference_in_conv2d_net(neural);
    
    // Check the result
    ptr[0] = mat_average_pool1;
    ptr[1] = mat_average_pool2;
    func_ptr formater = non_truncated;
    if (!check_activations(neural, formater)) {
        return false;
    }
    fprintf(stdout, ">>>>>>> Test: average pooling (unit strides): success...\n");
    
    //---------------------------------------
    // Test with non-unit strides
    //---------------------------------------
    
    free(nn->conv2d->conv_activations->val);
    free(nn->conv2d->conv_activations);
    
    sh = 2;
    maps_size[1] = 4;
    maps_size[2] = 2;
    set_up(neural, maps_size, 6, kh, sh, POOLING);
    
    // Apply the pooling
    inference_in_conv2d_net(neural);
    
    // Check the result
    ptr[0] = mat_average_pool_stride_1;
    ptr[1] = mat_average_pool_stride_2;
    if (!check_activations(neural, formater)) {
        return false;
    }
    fprintf(stdout, ">>>>>>> Test: average pooling (non-unit strides): success...\n");
    
    free(nn->conv2d->conv_activations->val);
    free(nn->conv2d->conv_activations);
    
    return true;
}

bool test_l2_pooling(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    //---------------------------------------
    // Test with unit strides
    //---------------------------------------
    
    unsigned int kh = 2;
    unsigned int sh = 1;
    
    // Assume a network with three layers: one feeding layer
    // and two max pooling layers
    nn->network_num_layers = 3;
    nn->conv2d->num_pooling_layers = 2;
    nn->conv2d->num_infer_ops = 2;
    int idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l > 0) {
            nn->conv2d->inferenceOps[idx] = l2_pooling_op;
            idx++;
        }
    }
    
    int maps_size[3] = {8, 7, 6};
    set_up(neural, maps_size, 6, kh, sh, POOLING);
    
    // Apply yje pooling
    inference_in_conv2d_net(neural);
    
    // Check the result
    ptr[0] = mat_l2_pool1;
    ptr[1] = mat_l2_pool2;
    func_ptr formater = truncated;
    if (!check_activations(neural, formater)) {
        return false;
    }
    fprintf(stdout, ">>>>>>> Test: L2 pooling (unit strides): success...\n");
    
    //---------------------------------------
    // Test with non-unit strides
    //---------------------------------------
    
    free(nn->conv2d->conv_activations->val);
    free(nn->conv2d->conv_activations);
    
    sh = 2;
    maps_size[1] = 4;
    maps_size[2] = 2;
    set_up(neural, maps_size, 6, kh, sh, POOLING);
    
    // Apply the pooling
    inference_in_conv2d_net(neural);
    
    // Check the result
    ptr[0] = mat_l2_pool_stride_1;
    ptr[1] = mat_l2_pool_stride_2;
    if (!check_activations(neural, formater)) {
        return false;
    }
    fprintf(stdout, ">>>>>>> Test: L2 pooling (non-unit strides): success...\n");
    
    free(nn->conv2d->conv_activations->val);
    free(nn->conv2d->conv_activations);
    
    return true;
}

bool test_convolution(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    
    unsigned int kh = 5;
    unsigned int sh = 1;
    
    // Assume a network with two layers: one feeding layer
    // and one convolution layer
    nn->network_num_layers = 2;
    nn->conv2d->num_conv2d_layers = 1;
    nn->conv2d->num_infer_ops = 1;
    int idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l > 0) {
            nn->conv2d->inferenceOps[idx] = infer_convolution_op;
            nn->conv2d->activationFunctions[idx] = activation_func;
            idx++;
        }
    }
    
    int maps_size[2] = {8, 4};
    set_up(neural, maps_size, 1, kh, sh, CONVOLUTION);
    
    // The weights tensors
    // t1 = shape[1,1,5,5]
    tensor_dict *dict = init_tensor_dict();
    dict->rank = 4;
    nn->conv2d->conv_weights = (tensor *)nn->conv2d->conv_weights_alloc(neural, (void *)dict, true);
    nn->conv2d->flipped_weights = (tensor *)nn->conv2d->conv_weights_alloc(neural, (void *)dict, false);
    
    // The flipping matrix
    // | 0  0  0  0  1  |
    // | 0  0  0  1  0  |
    // | 0  0  1  0  0  |
    // | 0  1  0  0  0  |
    // | 1  0  0  0  0  |
    dict->rank = 2;
    int vec[2] = {kh, kh};
    shape(dict->shape, nn->conv2d->num_conv2d_layers, dict->rank, vec);
    nn->conv2d->flip_matrices = (tensor *)nn->tensor(neural, *dict);
    nn->create_flip(neural);
    
    // Initialize the kernels (weights) matrices
    float kernel[5*5] = {0.1, 0.2, 0.3, 0.4, 0.5,
                         0.2, 0.3, 0.4, 0.5, 0.6,
                         0.3, 0.4, 0.5, 0.6, 0.7,
                         0.4, 0.5, 0.6, 0.7, 0.8,
                         0.9, 1.0, 1.1, 1.2, 1.3};
    
    int offset = 0;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        
        unsigned p = nn->conv2d->conv_weights->shape[l][0][0];
        unsigned q = nn->conv2d->conv_weights->shape[l][1][0];
        
        int stride1 = 0;
        for (int k=0; k<p; k++) {
            int stride2 = 0;
            for (int ll=0; ll<q; ll++) {
                int idx = 0;
                for (int i=0; i<kh; i++) {
                    for (int j=0; j<kh; j++) {
                        nn->conv2d->conv_weights->val[offset+(stride1+(stride2+((i*kh)+j)))] = kernel[idx];
                        idx++;
                    }
                }
                stride2 = stride2 + (kh * kh);
            }
            stride1 = stride1 + (q * kh * kh);
        }
        offset = offset + (p * q * kh * kh);
    }
    
    // The biases with zero values
    dict->rank = 1;
    nn->conv2d->conv_biases = (tensor *)nn->conv2d->conv_common_alloc(neural, (void *)dict, true);
    
    // The affine transformations
    dict->rank = 3;
    nn->conv2d->conv_affineTransformations = (tensor *)nn->conv2d->conv_common_alloc(neural, (void *)dict, true);
    
    // The convolution matrix tensors
    int maps[2] = {1,1};
    dict->rank = 4;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4] = {maps[l], maps[l+1],
                     maps_size[l+1] * maps_size[l+1],
                     maps_size[l] * maps_size[l]};
        shape(dict->shape, dict->rank, vec, l);
    }
    nn->conv2d->conv_matrices = (tensor *)nn->tensor(neural, *dict);
    
    // Flip the kernels
    nn->flip_kernels(neural);
    
    // Create the convolution matrices
    nn->conv_mat_update(neural);
    
    // The reference convolution
    float C[8*8];
    float flipped_kernel[5*5];
    float flip[5*5] = {0, 0, 0, 0, 1,
                       0, 0, 0, 1, 0,
                       0, 0, 1, 0 ,0,
                       0, 1, 0, 0, 0,
                       1, 0, 0, 0 ,0};
    flip_mat(kh, kh, kh, flip, kh, kernel, kh, flipped_kernel, kh);
    vDSP_f5x5(mat_feed, 8, 8, flipped_kernel, C);
    
    float ref_conv[4*4];
    idx = 0;
    int k = 0;
    for (int i=0; i<8; i++) {
        for (int j=0; j<8; j++) {
            if (C[idx] != 0.0f) {
                ref_conv[k] = truncated(C[idx]);
                k++;
            }
            idx++;
        }
    }
    
    // Apply the convolution
    inference_in_conv2d_net(neural);
    
    // Check the result
    ptr[0] = ref_conv;
    func_ptr formater = truncated;
    if (!check_activations(neural, formater)) {
        return false;
    }
    fprintf(stdout, ">>>>>>> Test: convolution: success...\n");
    
    free(nn->conv2d->conv_activations->val);
    free(nn->conv2d->conv_activations);
    
    free(nn->conv2d->conv_weights->val);
    free(nn->conv2d->conv_weights);
    
    free(nn->conv2d->flipped_weights->val);
    free(nn->conv2d->flipped_weights);
    
    free(nn->conv2d->flip_matrices->val);
    free(nn->conv2d->flip_matrices);
    
    free(nn->conv2d->conv_matrices->val);
    free(nn->conv2d->conv_matrices);
    
    free(nn->conv2d->conv_biases->val);
    free(nn->conv2d->conv_biases);
    
    free(nn->conv2d->conv_affineTransformations->val);
    free(nn->conv2d->conv_affineTransformations);
    
    free(dict);
    
    return true;
}

bool test_convolution_pooling(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int kh_c = 5;
    unsigned int kh_p = 2;
    unsigned int sh = 1;
    
    // Assume a network with three layers: one feeding layer,
    // one convolution layer and one pooling layer
    nn->network_num_layers = 3;
    nn->conv2d->num_conv2d_layers = 1;
    nn->conv2d->num_pooling_layers = 1;
    nn->conv2d->num_infer_ops = 2;
    int idx = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l == 1) {
            nn->conv2d->inferenceOps[idx] = infer_convolution_op;
            nn->conv2d->activationFunctions[idx] = activation_func;
            idx++;
        } else if (l == 2) {
            nn->conv2d->inferenceOps[idx] = max_pooling_op;
        }
    }
    
    int maps_size[3] = {8,4,3};
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l == 0) {
            nn->conv2d->parameters->topology[l][0] = FEED;
            nn->conv2d->parameters->topology[l][1] = 1;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
        } else if (l == 1) {
            nn->conv2d->parameters->topology[l][0] = CONVOLUTION;
            nn->conv2d->parameters->topology[l][1] = 1;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
            nn->conv2d->parameters->topology[l][4] = kh_c;
            nn->conv2d->parameters->topology[l][5] = kh_c;
            nn->conv2d->parameters->topology[l][6] = sh;
            nn->conv2d->parameters->topology[l][7] = sh;
        } else {
            nn->conv2d->parameters->topology[l][0] = POOLING;
            nn->conv2d->parameters->topology[l][1] = 1;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
            nn->conv2d->parameters->topology[l][4] = kh_p;
            nn->conv2d->parameters->topology[l][5] = kh_p;
            nn->conv2d->parameters->topology[l][6] = sh;
            nn->conv2d->parameters->topology[l][7] = sh;
        }
    }
    
    // Activation tensors:
    // a1 = shape[1,8,8]
    // a2 = shape[1,4,4]
    // a3 = shape[1,3,3]
    tensor_dict *dict = init_tensor_dict();
    dict->rank = 3;
    nn->conv2d->conv_activations = (tensor *)nn->conv2d->conv_activations_alloc(neural, (void *)dict, true);
    
    init_feed_activations(neural);
    
    // The weights tensors
    // t1 = shape[1,1,5,5]
    dict->rank = 4;
    nn->conv2d->conv_weights = (tensor *)nn->conv2d->conv_weights_alloc(neural, (void *)dict, true);
    nn->conv2d->flipped_weights = (tensor *)nn->conv2d->conv_weights_alloc(neural, (void *)dict, false);
    
    // The flipping matrix
    dict->rank = 2;
    int vec[2] = {kh_c, kh_c};
    shape(dict->shape, nn->conv2d->num_conv2d_layers, dict->rank, vec);
    nn->conv2d->flip_matrices = (tensor *)nn->tensor(neural, *dict);
    nn->create_flip(neural);
    
    init_convol_kernels(neural, kh_c);
    
    // The biases with zero values
    dict->rank = 1;
    nn->conv2d->conv_biases = (tensor *)nn->conv2d->conv_common_alloc(neural, (void *)dict, true);
    
    // The affine transformations
    dict->rank = 3;
    nn->conv2d->conv_affineTransformations = (tensor *)nn->conv2d->conv_common_alloc(neural, (void *)dict, true);
    
    // The convolution matrix tensors
    int maps[2] = {1,1};
    dict->rank = 4;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4] = {maps[l], maps[l+1],
                      maps_size[l+1] * maps_size[l+1],
                      maps_size[l] * maps_size[l]};
        shape(dict->shape, dict->rank, vec, l);
    }
    nn->conv2d->conv_matrices = (tensor *)nn->tensor(neural, *dict);
    
    // Flip the kernels
    nn->flip_kernels(neural);
    
    // Create the convolution matrices
    nn->conv_mat_update(neural);
    
    // The reference convolution
    float ref_conv[4*4];
    ref_convol(ref_conv, kh_c);
    
    // Apply the convolution and pooling
    inference_in_conv2d_net(neural);
    
    /// Check the result
    ptr[0] = ref_conv;
    ptr[1] = ref_pool;
    func_ptr formater = truncated;
    if (!check_activations(neural, formater)) {
        return false;
    }
    fprintf(stdout, ">>>>>>> Test: convolution->pooling: success...\n");
    
    free(nn->conv2d->conv_activations->val);
    free(nn->conv2d->conv_activations);
    
    free(nn->conv2d->conv_weights->val);
    free(nn->conv2d->conv_weights);
    
    free(nn->conv2d->flipped_weights->val);
    free(nn->conv2d->flipped_weights);
    
    free(nn->conv2d->flip_matrices->val);
    free(nn->conv2d->flip_matrices);
    
    free(nn->conv2d->conv_matrices->val);
    free(nn->conv2d->conv_matrices);
    
    free(nn->conv2d->conv_biases->val);
    free(nn->conv2d->conv_biases);
    
    free(nn->conv2d->conv_affineTransformations->val);
    free(nn->conv2d->conv_affineTransformations);
 
    free(dict);
    
    return true;
}

bool test_pooling_fully_connected(void * _Nonnull  neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int kh = 2;
    unsigned int sh = 2;
    
    // Assume a network with five layers: one feeding layer,
    // one pooling layer and three fully connected layers
    nn->network_num_layers = 5;
    nn->conv2d->num_pooling_layers = 1;
    nn->conv2d->num_dense_layers = 3;
    nn->conv2d->num_infer_ops = 4;
    int idx = 0;
    int k = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l == 1) {
            nn->conv2d->inferenceOps[idx] = max_pooling_op;
            idx++;
        } else if (l > 1) {
            nn->conv2d->inferenceOps[idx] = infer_fully_connected_op;
            nn->conv2d->activationFunctions[k] = activation_func;
            idx++;
            k++;
        }
    }
    
    int maps_size[5] = {8,4,4,4,4};
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l == 0) {
            nn->conv2d->parameters->topology[l][0] = FEED;
            nn->conv2d->parameters->topology[l][1] = 2;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
        } else if (l == 1) {
            nn->conv2d->parameters->topology[l][0] = POOLING;
            nn->conv2d->parameters->topology[l][1] = 2;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
            nn->conv2d->parameters->topology[l][4] = kh;
            nn->conv2d->parameters->topology[l][5] = kh;
            nn->conv2d->parameters->topology[l][6] = sh;
            nn->conv2d->parameters->topology[l][7] = sh;
        } else {
            nn->conv2d->parameters->topology[l][0] = FULLY_CONNECTED;
            nn->conv2d->parameters->topology[l][1] = maps_size[l];
        }
    }
    
    // Feeding and poolling layers activations
    // a1 = shape[2,8,8]
    // a2 = shape[2,4,4]
    tensor_dict *dict = init_tensor_dict();
    dict->rank = 3;
    nn->conv2d->conv_activations = (tensor *)nn->conv2d->conv_activations_alloc(neural, (void *)dict, true);
    
    init_feed_activations(neural);
    
    // Fully connected layers activations, biases and affine transformations
    dict->rank = 1;
    nn->conv2d->dense_activations = (tensor *)nn->conv2d->dense_common_alloc(neural, (void *)dict, true);
    nn->conv2d->dense_biases = (tensor *)nn->conv2d->dense_common_alloc(neural, (void *)dict, false);
    nn->conv2d->dense_affineTransformations = (tensor *)nn->conv2d->dense_common_alloc(neural, (void *)dict, false);
    
    // Fully connected layers weights
    dict->rank = 2;
    nn->conv2d->dense_weights = (tensor *)nn->conv2d->dense_weights_alloc(neural, (void *)dict, true);
    init_dense_weights(neural);
    
    // Calculate the reference result
    float vec[(4*4)*2];
    idx = 0;
    for (int l=0; l<2; l++) {
        for (int i=0; i<4*4; i++) {
            vec[idx] = mat_max_pool_stride_1[i];
            idx++;
        }
    }

    int offset = 0;
    float result[4];
    for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
        
        unsigned int m = nn->conv2d->dense_weights->shape[l][0][0];
        unsigned int n = nn->conv2d->dense_weights->shape[l][1][0];
        
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, nn->conv2d->dense_weights->val+offset, n, vec, 1, 0.0f, result, 1);
        memcpy(vec, result, m*sizeof(float));
        offset = offset + (m * n);
    }
    
    // Apply the inference
    inference_in_conv2d_net(neural);
    
    // Check the result
    offset = 0;
    for (int l=0; l<nn->conv2d->num_dense_layers-1; l++) {
        offset = offset + nn->conv2d->dense_activations->shape[l][0][0];
    }
    for (int i=0; i<4; i++) {
        if (nn->conv2d->dense_activations->val[offset+i] != result[i]) {
            return false;
        }
    }
    fprintf(stdout, ">>>>>>> Test: pooling->fully connected: success...\n");
    
    free(nn->conv2d->conv_activations->val);
    free(nn->conv2d->conv_activations);
    
    free(nn->conv2d->dense_activations->val);
    free(nn->conv2d->dense_activations);
    free(nn->conv2d->dense_biases->val);
    free(nn->conv2d->dense_biases);
    free(nn->conv2d->dense_affineTransformations->val);
    free(nn->conv2d->dense_affineTransformations);
    
    free(nn->conv2d->dense_weights->val);
    free(nn->conv2d->dense_weights);
    
    free(dict);
    
    return true;
}

bool test_dummy_convol_net(void * _Nonnull neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int kh_c = 5;
    unsigned int kh_p = 2;
    unsigned int sh = 1;
    
    // Assume a network with six layers: one feeding layer,
    // one convolution layer, one pooling layer and three
    // fully connected layers
    nn->network_num_layers = 6;
    nn->conv2d->num_conv2d_layers = 1;
    nn->conv2d->num_pooling_layers = 1;
    nn->conv2d->num_dense_layers = 3;
    nn->conv2d->num_infer_ops = 5;
    int idx = 0;
    int k = 0;
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l == 1) {
            nn->conv2d->inferenceOps[idx] = infer_convolution_op;
            nn->conv2d->activationFunctions[k] = activation_func;
            idx++;
            k++;
        } else if (l == 2) {
            nn->conv2d->inferenceOps[idx] = max_pooling_op;
            idx++;
        } else if (l > 2) {
            nn->conv2d->inferenceOps[idx] = infer_fully_connected_op;
            nn->conv2d->activationFunctions[k] = activation_func;
            idx++;
            k++;
        }
    }
    
    int maps_size[6] = {8,4,3,4,4,4};
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l == 0) {
            nn->conv2d->parameters->topology[l][0] = FEED;
            nn->conv2d->parameters->topology[l][1] = 2;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
        } else if (l == 1) {
            nn->conv2d->parameters->topology[l][0] = CONVOLUTION;
            nn->conv2d->parameters->topology[l][1] = 1;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
            nn->conv2d->parameters->topology[l][4] = kh_c;
            nn->conv2d->parameters->topology[l][5] = kh_c;
            nn->conv2d->parameters->topology[l][6] = sh;
            nn->conv2d->parameters->topology[l][7] = sh;
        } else if (l == 2) {
            nn->conv2d->parameters->topology[l][0] = POOLING;
            nn->conv2d->parameters->topology[l][1] = 1;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
            nn->conv2d->parameters->topology[l][4] = kh_p;
            nn->conv2d->parameters->topology[l][5] = kh_p;
            nn->conv2d->parameters->topology[l][6] = sh;
            nn->conv2d->parameters->topology[l][7] = sh;
        } else {
            nn->conv2d->parameters->topology[l][0] = FULLY_CONNECTED;
            nn->conv2d->parameters->topology[l][1] = maps_size[l];
        }
    }
    
    // Feeding, convolution and pooling layers activations
    // a1 = shape[1,8,8]
    // a2 = shape[1,4,4]
    // a2 = shape[1,3,3]
    tensor_dict *dict = init_tensor_dict();
    dict->rank = 3;
    nn->conv2d->conv_activations = (tensor *)nn->conv2d->conv_activations_alloc(neural, (void *)dict, true);
    
    init_feed_activations(neural);
    
    // The weights tensors
    // t1 = shape[1,1,5,5]
    dict->rank = 4;
    nn->conv2d->conv_weights = (tensor *)nn->conv2d->conv_weights_alloc(neural, (void *)dict, true);
    nn->conv2d->flipped_weights = (tensor *)nn->conv2d->conv_weights_alloc(neural, (void *)dict, false);
    
    // The flipping matrix
    dict->rank = 2;
    int vec1[2] = {kh_c, kh_c};
    shape(dict->shape, nn->conv2d->num_conv2d_layers, dict->rank, vec1);
    nn->conv2d->flip_matrices = (tensor *)nn->tensor(neural, *dict);
    nn->create_flip(neural);
    
    init_convol_kernels(neural, kh_c);
    
    // The convolution biases with zero values
    dict->rank = 1;
    nn->conv2d->conv_biases = (tensor *)nn->conv2d->conv_common_alloc(neural, (void *)dict, true);
    
    // The convolution affine transformations
    dict->rank = 3;
    nn->conv2d->conv_affineTransformations = (tensor *)nn->conv2d->conv_common_alloc(neural, (void *)dict, true);
    
    // The convolution matrix tensors
    int maps[2] = {1,1};
    dict->rank = 4;
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4] = {maps[l], maps[l+1],
            maps_size[l+1] * maps_size[l+1],
            maps_size[l] * maps_size[l]};
        shape(dict->shape, dict->rank, vec, l);
    }
    nn->conv2d->conv_matrices = (tensor *)nn->tensor(neural, *dict);
    
    // Flip the kernels
    nn->flip_kernels(neural);
    
    // Create the convolution matrices
    nn->conv_mat_update(neural);
    
    // Fully connected layers activations, biases and affine transformations
    dict->rank = 2;
    nn->conv2d->dense_activations = (tensor *)nn->conv2d->dense_common_alloc(neural, (void *)dict, true);
    nn->conv2d->dense_biases = (tensor *)nn->conv2d->dense_common_alloc(neural, (void *)dict, false);
    nn->conv2d->dense_affineTransformations = (tensor *)nn->conv2d->dense_common_alloc(neural, (void *)dict, false);
    
    // Fully connected layers weights
    dict->rank = 2;
    nn->conv2d->dense_weights = (tensor *)nn->conv2d->dense_weights_alloc(neural, (void *)dict, true);
    init_dense_weights(neural);
    
    // Calculate the reference result
    float vec2[(3*3)*2];
    idx = 0;
    for (int l=0; l<2; l++) {
        for (int i=0; i<3*3; i++) {
            vec2[idx] = ref_pool[i];
            idx++;
        }
    }
    
    int offset = 0;
    float result[4];
    for (int l=0; l<nn->conv2d->num_dense_layers; l++) {
        
        unsigned int m = nn->conv2d->dense_weights->shape[l][0][0];
        unsigned int n = nn->conv2d->dense_weights->shape[l][1][0];
        
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, nn->conv2d->dense_weights->val+offset, n, vec2, 1, 0.0f, result, 1);
        memcpy(vec2, result, m*sizeof(float));
        offset = offset + (m * n);
    }
    
    // The reference convolution
    float ref_conv[4*4];
    ref_convol(ref_conv, kh_c);
    
    // Apply the inference
    inference_in_conv2d_net(neural);
    
    ptr[0] = ref_conv;
    ptr[1] = ref_pool;
    func_ptr formater = truncated;
    if (!check_activations(neural, formater)) {
        return false;
    }
    
    // Check the result
    offset = 0;
    for (int l=0; l<nn->conv2d->num_dense_layers-1; l++) {
        offset = offset + nn->conv2d->dense_activations->shape[l][0][0];
    }
    for (int i=0; i<4; i++) {
        if (nn->conv2d->dense_activations->val[offset+i] != result[i]) {
            return false;
        }
    }
    fprintf(stdout, ">>>>>>> Test: dummy convol net: success...\n");
    
    free(nn->conv2d->conv_activations->val);
    free(nn->conv2d->conv_activations);
    
    free(nn->conv2d->conv_weights->val);
    free(nn->conv2d->conv_weights);
    
    free(nn->conv2d->flipped_weights->val);
    free(nn->conv2d->flipped_weights);
    
    free(nn->conv2d->flip_matrices->val);
    free(nn->conv2d->flip_matrices);
    
    free(nn->conv2d->conv_matrices->val);
    free(nn->conv2d->conv_matrices);
    
    free(nn->conv2d->conv_biases->val);
    free(nn->conv2d->conv_biases);
    
    free(nn->conv2d->conv_affineTransformations->val);
    free(nn->conv2d->conv_affineTransformations);
    
    free(nn->conv2d->dense_activations->val);
    free(nn->conv2d->dense_activations);
    free(nn->conv2d->dense_biases->val);
    free(nn->conv2d->dense_biases);
    free(nn->conv2d->dense_affineTransformations->val);
    free(nn->conv2d->dense_affineTransformations);
    
    free(nn->conv2d->dense_weights->val);
    free(nn->conv2d->dense_weights);
    
    free(dict);
    
    return true;
}

int main(int argc, const char * argv[]) {
    
    BrainStormNet *neural = new_dense_net();
    
    // Test tensors creation and initialization
    if (!test_create_tensors(neural)) {
        fprintf(stderr, "Test: create tensor: failed.\n");
        return -1;
    }
    free(neural);
    
    // Test kernels flipping
    neural = new_conv2d_net();
    if (!test_kernels_flipping(neural)) {
        fprintf(stderr, "Test: tensor flipping: failed.\n");
        return -1;
    }
    free(neural);
    
    // Test the convolution matrix creation
    neural = new_conv2d_net();
    if (!test_create_convol_matrix(neural)) {
        fprintf(stderr, "Test: convolution matrix: failed.\n");
        return -1;
    }
    free(neural);
    
    // Test max pooling
    neural = new_conv2d_net();
    if (!test_max_pooling(neural)) {
        fprintf(stderr, "Test: max pooling: failed.\n");
        return -1;
    }
    free(neural);
    
    // Test average pooling
    neural = new_conv2d_net();
    if (!test_average_pooling(neural)) {
        fprintf(stderr, "Test: average pooling: failed.\n");
        return -1;
    }
    free(neural);
    
    // Test L2 pooling
    neural = new_conv2d_net();
    if (!test_l2_pooling(neural)) {
        fprintf(stderr, "Test: L2 pooling: failed.\n");
        return -1;
    }
    free(neural);
    
    // Test the convolution operation
    neural = new_conv2d_net();
    if (!test_convolution(neural)) {
        fprintf(stderr, "Test: convolution: failed.\n");
        return -1;
    }
    free(neural);
    
    // Test the convolution->pooling operations
    neural = new_conv2d_net();
    if (!test_convolution_pooling(neural)) {
        fprintf(stderr, "Test: convolution->pooling: failed.\n");
        return -1;
    }
    free(neural);
    
    // Test the pooling->fully connected operations
    neural = new_conv2d_net();
    if (!test_pooling_fully_connected(neural)) {
        fprintf(stderr, "Test: pooling->fully connected: failed.\n");
        return -1;
    }
    
    // Test a dummy convolution network with one convolution
    // layer, one pooling layer and three fully connected layers
    neural = new_conv2d_net();
    if (!test_dummy_convol_net(neural)) {
        fprintf(stderr, "Test: dummy convol net: failed.\n");
        return -1;
    }
    free(neural);
    
    fprintf(stdout, "All tests passed.\n");
    return 0;
}
