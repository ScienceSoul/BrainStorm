//
//  NetworkUtils.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 26/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef NetworkUtils_h
#define NetworkUtils_h

#include <stdbool.h>
#include "Utils.h"

typedef struct tensor {
    float * _Nullable val;
    unsigned int shape[MAX_NUMBER_NETWORK_LAYERS][MAX_TENSOR_RANK][1];
    unsigned int rank;
} tensor;

typedef struct tensor_dict {
    bool init;
    bool full_connected;
    unsigned int flattening_length;
    unsigned int shape[MAX_NUMBER_NETWORK_LAYERS][MAX_TENSOR_RANK][1];
    unsigned int rank;
} tensor_dict;

void standard_normal_initializer(void * _Nonnull neural, void * _Nonnull kernel, int l, int offset);
void xavier_he_initializer(void * _Nonnull neural, void * _Nonnull kernel, int l, int offset);

void * _Nonnull tensor_create_(void * _Nonnull self, tensor_dict tensor_dict);
void * _Nonnull tensor_create(void * _Nonnull self, tensor_dict tensor_dict);

int loadParametersFromImputFile(void * _Nonnull self, const char * _Nonnull paraFile);

#endif /* NetworkUtils_h */
