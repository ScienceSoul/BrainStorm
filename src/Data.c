//
//  Data.c
//  FeedforwardNT
//
//  Created by Hakime Seddik on 26/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#include <stdio.h>
#include "NeuralNetwork.h"
#include "Memory.h"

static float * _Nonnull * _Nonnull createTrainigData(float * _Nonnull * _Nonnull dataSet, unsigned int start, unsigned int end, unsigned int * _Nonnull t1, unsigned int * _Nonnull t2, int * _Nonnull classifications, unsigned int numberOfClassifications, int * _Nonnull ntLayers, int numberOfLayers) {
    
    float **trainingData = NULL;
    trainingData = floatmatrix(0, end-1, 0, (ntLayers[0]+ntLayers[numberOfLayers-1])-1);
    *t1 = end;
    *t2 = ntLayers[0]+ntLayers[numberOfLayers-1];
    
    if (ntLayers[numberOfLayers-1] != numberOfClassifications) {
        fatal(PROGRAM_NAME, "the number of classifications should be equal to the number of activations at the output layer.");
    }
    
    for (int i=0; i<end; i++) {
        for (int j=0; j<ntLayers[0]; j++) {
            trainingData[i][j] = dataSet[i][j];
        }
        
        // Binarization of the input ground-truth to get a one-hot-vector
        for (int k=0; k<numberOfClassifications; k++) {
            if (dataSet[i][ntLayers[0]] == (float)classifications[k]) {
                trainingData[i][ntLayers[0]+k] = 1.0f;
            } else trainingData[i][ntLayers[0]+k] = 0.0f;
        }
    }
    
    return trainingData;
}

static float * _Nonnull * _Nonnull getData(float * _Nonnull * _Nonnull dataSet, unsigned int len1, unsigned int len2, unsigned int start, unsigned int end, unsigned int * _Nonnull t1, unsigned int * _Nonnull t2) {
    
    float **data = floatmatrix(0, end, 0, len2-1);
    *t1 = end;
    *t2 = len2;
    
    int idx = 0;
    for (int i=(int)start; i<start+end; i++) {
        for (int j=0; j<len2; j++) {
            data[idx][j] = dataSet[i][j];
        }
        idx++;
    }
    return data;
}

void loadData(void * _Nonnull self, const char * _Nonnull dataSetName, const char * _Nonnull trainFile, const char * _Nonnull testFile, bool testData) {
    
    unsigned int len1=0, len2=0;
    float **raw_training = NULL;
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    fprintf(stdout, "%s: load the <%s> training data set.\n", PROGRAM_NAME, dataSetName);
    raw_training = nn->data->training->reader(trainFile, &len1, &len2);
    shuffle(raw_training, len1, len2);
    
    nn->data->training->set = createTrainigData(raw_training, 0, nn->parameters->split[0], &nn->data->training->m, &nn->data->training->n, nn->parameters->classifications, nn->parameters->numberOfClassifications, nn->parameters->topology, nn->parameters->numberOfLayers);
    
    if (testData) {
        nn->data->test->set = nn->data->test->reader(testFile, &nn->data->test->m, &nn->data->test->n);
        nn->data->validation->set = getData(raw_training, len1, len2, nn->parameters->split[0], nn->parameters->split[1], &nn->data->validation->m, &nn->data->validation->n);
    } else {
        nn->data->test->set = getData(raw_training, len1, len2, nn->parameters->split[0], nn->parameters->split[1], &nn->data->test->m, &nn->data->test->n);
    }
}
