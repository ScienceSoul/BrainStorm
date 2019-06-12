//
//  Data.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 26/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef Data_h
#define Data_h

#include <stdbool.h>

typedef struct training {
    void * _Nullable set;
    void * _Nullable labels;
    float * _Nullable * _Nullable (* _Nullable reader)(const char * _Nonnull fileName, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2, unsigned int * _Nonnull num_channels);
} training;
typedef struct test {
    void * _Nullable set;
    void * _Nullable labels;
    float * _Nullable * _Nullable (* _Nullable reader)(const char * _Nonnull fileName, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2, unsigned int * _Nonnull num_channels);
} test;
typedef struct validation {
    void * _Nullable set;
    void * _Nullable labels;
} validation;

typedef struct data {
    training * _Nullable training;
    test * _Nullable test;
    validation *_Nullable validation;
    void (* _Nullable init)(void * _Nonnull self);
    void (* _Nullable load)(void * _Nonnull self, const char * _Nonnull dataSetName, const char * _Nonnull trainFile, const char * _Nonnull testFile, bool testData, bool binarization);
} data;

void load_data(void * _Nonnull self, const char * _Nonnull dataSetName, const char * _Nonnull trainFile, const char * _Nonnull testFile, bool testData, bool binarization);

#endif /* Data_h */
