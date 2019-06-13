//
//  NetworkOps.h
//  BrainStorm
//
//  Created by Hakime Seddik on 18/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef NetworkOps_h
#define NetworkOps_h

typedef struct progress_dict {
    unsigned int batch_size;
    unsigned int percent;
} progress_dict;

typedef void (* _Nullable ptr_inference_func)(void * _Nonnull self);
typedef void (* _Nullable ptr_backpropag_func)(void * _Nonnull self,
                                       void (* _Nullable ptr_inference_func)(void * _Nonnull self));
typedef void (* _Nullable ptr_batch_accumul_func)(void * _Nonnull self);

void mini_batch_loop(void * _Nonnull neural, unsigned int batch_size,
                   ptr_inference_func inference, ptr_backpropag_func backpropagation, ptr_batch_accumul_func batch_accumulation);

void next_batch(void * _Nonnull neural, tensor * _Nonnull features, tensor * _Nonnull labels, unsigned int batch_size, int * _Nullable remainder, bool do_remainder);
int batch_range(void * _Nonnull neural, unsigned int batch_size, int * _Nullable remainder);

void progression(void * _Nonnull neural, progress_dict progress_dict);

float math_ops(float * _Nonnull vector, unsigned int n, char * _Nonnull op);

void eval_prediction(void * _Nonnull self, char * _Nonnull data_set, float * _Nonnull out, bool metal);
float eval_cost(void * _Nonnull self, char * _Nonnull data_set, bool binarization);

void flip_kernels(void * _Nonnull neural);
void flip_deltas(void * _Nonnull neural, unsigned int q, unsigned int fh, unsigned int fw);
void kernel_mat_update(void * _Nonnull neural);

// Only used when loading a network from a param file
void train_loop(void * _Nonnull  neural);

#endif /* NetworkOps_h */
