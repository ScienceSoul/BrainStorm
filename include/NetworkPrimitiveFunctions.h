//
//  NetworkPrimitiveFunctions.h
//  BrainStorm
//
//  Created by Hakime Seddik on 12/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef NetworkPrimitiveFunctions_h
#define NetworkPrimitiveFunctions_h

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

// Only used when loading a network from a param file
void trainLoop(void * _Nonnull  neural);

#endif /* NetworkPrimitiveFunctions_h */
