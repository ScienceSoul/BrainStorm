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

void miniBatchLoop(void * _Nonnull neural, unsigned int batch_size,
                   ptr_inference_func inference, ptr_backpropag_func backpropagation, ptr_batch_accumul_func batch_accumulation);

void nextBatch(void * _Nonnull neural, float * _Nonnull * _Nonnull placeholder, unsigned int batchSize);
int batchRange(void * _Nonnull neural, unsigned int batchSize);

void progression(void * _Nonnull neural, progress_dict progress_dict);

float mathOps(float * _Nonnull vector, unsigned int n, char * _Nonnull op);

void evalPrediction(void * _Nonnull self, char * _Nonnull dataSet, float * _Nonnull out, bool metal);
float evalCost(void * _Nonnull self, char * _Nonnull dataSet, bool binarization);

void flipKernels(void * _Nonnull neural);
void flipDeltas(void * _Nonnull neural, unsigned int q, unsigned int fh, unsigned int fw);
void convMatUpdate(void * _Nonnull neural);

// Only used when loading a network from a param file
void trainLoop(void * _Nonnull  neural);

#endif /* NetworkOps_h */
