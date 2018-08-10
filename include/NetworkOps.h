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

void feedforward(void * _Nonnull self);
void backpropagation(void * _Nonnull self);
void batchAccumulation(void * _Nonnull self);
void miniBatchLoop(void * _Nonnull neural, unsigned int batch_size);

void nextBatch(void * _Nonnull neural, float * _Nonnull * _Nonnull placeholder, unsigned int batchSize);
int batchRange(void * _Nonnull neural, unsigned int batchSize);

void progression(void * _Nonnull neural, progress_dict progress_dict);

float mathOps(float * _Nonnull vector, unsigned int n, char * _Nonnull op);

void evalPrediction(void * _Nonnull self, char * _Nonnull dataSet, float * _Nonnull out, bool metal);
float evalCost(void * _Nonnull self, char * _Nonnull dataSet, bool binarization);

void maxPool(void);
void l2Pool(void);
void averagePool(void);

// Only used when loading a network from a param file
void trainLoop(void * _Nonnull  neural);

#endif /* NetworkOps_h */
