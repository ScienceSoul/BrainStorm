//
//  main.c
//  BrainStormTest
//
//  Created by Hakime Seddik on 18/09/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

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
float mat_pool_feed[8*8] = {1,  2,  3,  4,  5,  6,  7,  8,
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

// The array of pointers to pooling results
float *ptr[2];

void set_up_pooling(void * _Nonnull neural, int * _Nonnull maps_size, unsigned int number_of_maps, unsigned int kh, unsigned int sh ) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    for (int l=0; l<nn->network_num_layers; l++) {
        if (l == 0) {
            nn->conv2d->parameters->topology[l][0] = FEED;
            nn->conv2d->parameters->topology[l][1] = number_of_maps;
            nn->conv2d->parameters->topology[l][2] = maps_size[l];
            nn->conv2d->parameters->topology[l][3] = maps_size[l];
        } else {
            nn->conv2d->parameters->topology[l][0] = POOLING;
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
    // a1 = shape[number_of_maps,maps_size[l],maps_size[l]] = shape[6,8,8]
    // a2 = shape[number_of_maps,maps_size[l],maps_size[l]] = shape[6,7,7]
    // a3 = shape[number_of_maps,maps_size[l],maps_size[l]] = shape[6,6,6]
    tensor_dict dict;
    dict.rank = 3;
    for (int l=0; l<nn->network_num_layers; l++) {
        int vec[3];
        vec[0] = nn->conv2d->parameters->topology[l][1];
        vec[1] = nn->conv2d->parameters->topology[l][2];
        vec[2] = nn->conv2d->parameters->topology[l][3];
        shape(dict.shape, dict.rank, vec, l);
    }
    dict.flattening_length =  nn->network_num_layers;
    nn->conv2d->conv_activations = (tensor *)nn->tensor(neural, dict);
    
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
                    nn->conv2d->conv_activations->val[stride+(i*fw+j)] = mat_pool_feed[idx];
                    idx++;
                }
            }
            stride = stride + (fh * fw);
        }
    }
}

bool check_pooling(void * _Nonnull neural) {
    
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
                        if (nn->conv2d->conv_activations->val[offset+(stride+((i*fw)+j))] != *(ptr[p_idx]+idx)) {
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

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

bool test_create_tensors(void * neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int kh = 5;
    unsigned int layers = 10;
    float init_value = 1.0f;
    int vec[5];
    
    tensor_dict dict;
    dict.flattening_length = layers;
    dict.init_with_value = true;
    dict.init_value = init_value;
    
    dict.rank = 1;
    vec[0] = kh;
    shape(dict.shape, layers, dict.rank, vec);
    tensor *tensor_1d = (tensor *)nn->tensor(neural, dict);
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
    
    dict.rank = 2;
    vec[1] = kh;
    shape(dict.shape, layers, dict.rank, vec);
    tensor *tensor_2d = (tensor *)nn->tensor(neural, dict);
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
    
    dict.rank = 3;
    vec[2] = kh;
    shape(dict.shape, layers, dict.rank, vec);
    tensor *tensor_3d = (tensor *)nn->tensor(neural, dict);
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
    
    dict.rank = 4;
    vec[3] = kh;
    shape(dict.shape, layers, dict.rank, vec);
    tensor *tensor_4d = (tensor *)nn->tensor(neural, dict);
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
    
    dict.rank = 5;
    vec[4] = kh;
    shape(dict.shape, layers, dict.rank, vec);
    tensor *tensor_5d = (tensor *)nn->tensor(neural, dict);
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
    
    return true;
}

bool test_kernels_flipping(void *neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    unsigned int kh = 5;
    
    // The weights tensors
    // t1 = shape[1,6,5,5]
    // t2 = shape[6,12,5,5]
    nn->conv2d->num_conv2d_layers = 2;
    tensor_dict dict;
    dict.rank = 4;
    int maps[4] = {1, 6, 12};
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4];
        vec[0] = maps[l];
        vec[1] = maps[l+1];
        vec[2] = kh;
        vec[3] = kh;
        shape(dict.shape, dict.rank, vec, l);
    }
    dict.flattening_length = nn->conv2d->num_conv2d_layers;
    nn->conv2d->conv_weights = (tensor *)nn->tensor(neural, dict);
    nn->conv2d->flipped_weights = (tensor *)nn->tensor(neural, dict);
    
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
    dict.rank = 2;
    int vec[2] = {kh, kh};
    shape(dict.shape, nn->conv2d->num_conv2d_layers, dict.rank, vec);
    nn->conv2d->flip_matrices = (tensor *)nn->tensor(neural, dict);
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
    
    fprintf(stdout, ">>>>>>> Test: tensor flipping: success...\n");
    return true;
}

bool test_create_convol_matrix(void *neural) {
    
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
    tensor_dict dict;
    dict.rank = 4;
    int maps[4] = {1, 6, 12};
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4];
        vec[0] = maps[l];
        vec[1] = maps[l+1];
        vec[2] = kh;
        vec[3] = kh;
        shape(dict.shape, dict.rank, vec, l);
    }
    dict.flattening_length = nn->conv2d->num_conv2d_layers;
    nn->conv2d->conv_weights = (tensor *)nn->tensor(neural, dict);
    nn->conv2d->flipped_weights = (tensor *)nn->tensor(neural, dict);
    
    dict.rank = 2;
    int vec[2] = {kh, kh};
    shape(dict.shape, nn->conv2d->num_conv2d_layers, dict.rank, vec);
    nn->conv2d->flip_matrices = (tensor *)nn->tensor(neural, dict);
    nn->create_flip(neural);
    
    // The convolution matrix tensors
    // Assume 4 x 4 input layer. With 2 x 2 kernels and unit strides,
    // after the first convolution we get maps of size 3 x 3. After the
    // second convolution we get maps of size 2 x 2
    // The convolution matrix tensors are:
    //   c1 = shape[1,6,9,16]
    //   c2 = shape[6,12,4,9]
    dict.rank = 4;
    int maps_size[3] = {4, 3, 2};
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4];
        vec[0] = maps[l];
        vec[1] = maps[l+1];
        vec[2] = maps_size[l+1] * maps_size[l+1];
        vec[3] = maps_size[l] * maps_size[l];
        shape(dict.shape, dict.rank, vec, l);
    }
    nn->conv2d->conv_matrices = (tensor *)nn->tensor(neural, dict);
    
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
    dict.rank = 4;
    int map_size_stride[2] = {4, 2};
    for (int l=0; l<nn->conv2d->num_conv2d_layers; l++) {
        int vec[4];
        vec[0] = maps[l];
        vec[1] = maps[l+1];
        vec[2] = map_size_stride[l+1] * map_size_stride[l+1];
        vec[3] = map_size_stride[l] * map_size_stride[l];
        shape(dict.shape, dict.rank, vec, l);
    }
    nn->conv2d->conv_matrices = (tensor *)nn->tensor(neural, dict);
    
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
    
    return true;
}

bool test_max_pooling(void *neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    //---------------------------------------
    // Test with unit strides
    //---------------------------------------
    
    // Assume a network with three layers: one feeding layer
    // and two max pooling layers
    unsigned int kh = 2;
    unsigned int sh = 1;
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
    set_up_pooling(neural, maps_size, 6, kh, sh);
    
    // Apply the pooling
    inference_in_conv2d_net(neural);
    
    // Check the result
    ptr[0] = mat_max_pool1;
    ptr[1] = mat_max_pool2;
    if (!check_pooling(neural)) {
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
    set_up_pooling(neural, maps_size, 6, kh, sh);
    
    // Apply the pooling
    inference_in_conv2d_net(neural);
    
     // Check the result
    ptr[0] = mat_max_pool_stride_1;
    ptr[1] = mat_max_pool_stride_2;
    if (!check_pooling(neural)) {
        return false;
    }
    fprintf(stdout, ">>>>>>> Test: max pooling (non-unit strides): success...\n");
    
    return true;
}

bool test_average_pooling(void *neural) {
    
    BrainStormNet *nn = (BrainStormNet *)neural;
    
    //---------------------------------------
    // Test with unit strides
    //---------------------------------------
    
    // Assume a network with three layers: one feeding layer
    // and two max pooling layers
    unsigned int kh = 2;
    unsigned int sh = 1;
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
    set_up_pooling(neural, maps_size, 6, kh, sh);
    
    // Apply the pooling
    inference_in_conv2d_net(neural);
    
    // Check the result
    ptr[0] = mat_average_pool1;
    ptr[1] = mat_average_pool2;
    if (!check_pooling(neural)) {
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
    set_up_pooling(neural, maps_size, 6, kh, sh);
    
    // Apply the pooling
    inference_in_conv2d_net(neural);
    
    // Check the result
    ptr[0] = mat_average_pool_stride_1;
    ptr[1] = mat_average_pool_stride_2;
    if (!check_pooling(neural)) {
        return false;
    }
    fprintf(stdout, ">>>>>>> Test: average pooling (non-unit strides): success...\n");
    
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
    
    fprintf(stdout, "All tests passed.\n");
    return 0;
}
