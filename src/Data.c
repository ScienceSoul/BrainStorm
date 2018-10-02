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

static float __attribute__((overloadable)) * _Nonnull * _Nonnull createTrainigData(float * _Nonnull * _Nonnull dataSet, unsigned int start, unsigned int end, unsigned int * _Nonnull t1, unsigned int * _Nonnull t2, int * _Nonnull classifications, unsigned int numberOfClassifications, int * _Nonnull topology, int numberOfLayers) {
    
    float **trainingData = NULL;
    trainingData = floatmatrix(0, end-1, 0, (topology[0]+topology[numberOfLayers-1])-1);
    *t1 = end;
    *t2 = topology[0]+topology[numberOfLayers-1];
    
    if (topology[numberOfLayers-1] != numberOfClassifications) {
        fatal(DEFAULT_CONSOLE_WRITER, "the number of classifications should be equal to the number of activations at the output layer.");
    }
    
    for (int i=0; i<end; i++) {
        for (int j=0; j<topology[0]; j++) {
            trainingData[i][j] = dataSet[i][j];
        }
        
        // Binarization of the input ground-truth to get a one-hot-vector
        for (int k=0; k<numberOfClassifications; k++) {
            if (dataSet[i][topology[0]] == (float)classifications[k]) {
                trainingData[i][topology[0]+k] = 1.0f;
            } else trainingData[i][topology[0]+k] = 0.0f;
        }
    }
    return trainingData;
}

static float __attribute__((overloadable)) * _Nonnull * _Nonnull createTrainigData(float * _Nonnull * _Nonnull dataSet, unsigned int start, unsigned int end, unsigned int * _Nonnull t1, unsigned int * _Nonnull t2, int * _Nonnull classifications, unsigned int numberOfClassifications, int topology[][8], int numberOfLayers) {
    
    //TODO: Take into accounts channels > 1
    
    float **trainingData = NULL;
    int n = (topology[0][2]*topology[0][3]) + topology[numberOfLayers-1][1];
    trainingData = floatmatrix(0, end-1, 0, n-1);
    *t1 = end;
    *t2 = n;
    
    if (topology[numberOfLayers-1][1] != numberOfClassifications) {
        fatal(DEFAULT_CONSOLE_WRITER, "the number of classifications should be equal to the number of activations at the output layer.");
    }
    
    for (int i=0; i<end; i++) {
        for (int j=0; j<(topology[0][2]*topology[0][3]); j++) {
            trainingData[i][j] = dataSet[i][j];
        }
        
        // Binarization of the input ground-truth to get a one-hot-vector
        for (int k=0; k<numberOfClassifications; k++) {
            if (dataSet[i][(topology[0][2]*topology[0][3])] == (float)classifications[k]) {
                trainingData[i][(topology[0][2]*topology[0][3])+k] = 1.0f;
            } else trainingData[i][(topology[0][2]*topology[0][3])+k] = 0.0f;
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
    
    BrainStormNet *nn = (BrainStormNet *)self;
    
    fprintf(stdout, "%s: load the <%s> training data set...\n", DEFAULT_CONSOLE_WRITER, dataSetName);
    raw_training = nn->data->training->reader(trainFile, &len1, &len2);
    shuffle(raw_training, len1, len2);
    
    if (nn->is_dense_network) {
        nn->data->training->set = createTrainigData(raw_training, 0, nn->dense->parameters->split[0], &nn->data->training->m, &nn->data->training->n, nn->dense->parameters->classifications, nn->dense->parameters->numberOfClassifications, nn->dense->parameters->topology, nn->network_num_layers);
    } else if (nn->is_conv2d_network) {
        nn->data->training->set = createTrainigData(raw_training, 0, nn->conv2d->parameters->split[0], &nn->data->training->m, &nn->data->training->n, nn->conv2d->parameters->classifications, nn->conv2d->parameters->numberOfClassifications, nn->conv2d->parameters->topology, nn->network_num_layers);
    }
    
    if (testData) {
        fprintf(stdout, "%s: load test data set in <%s>...\n", DEFAULT_CONSOLE_WRITER, dataSetName);
        nn->data->test->set = nn->data->test->reader(testFile, &nn->data->test->m, &nn->data->test->n);
        if (nn->is_dense_network) {
            nn->data->validation->set = getData(raw_training, len1, len2, nn->dense->parameters->split[0], nn->dense->parameters->split[1], &nn->data->validation->m, &nn->data->validation->n);
        } else if (nn->is_conv2d_network) {
            nn->data->validation->set = getData(raw_training, len1, len2, nn->conv2d->parameters->split[0], nn->conv2d->parameters->split[1], &nn->data->validation->n, &nn->data->validation->n);
        }
    } else {
        if (nn->is_dense_network) {
            nn->data->test->set = getData(raw_training, len1, len2, nn->dense->parameters->split[0], nn->dense->parameters->split[1], &nn->data->test->m, &nn->data->test->n);
        } else if (nn->is_conv2d_network) {
            nn->data->test->set = getData(raw_training, len1, len2, nn->conv2d->parameters->split[0], nn->conv2d->parameters->split[1], &nn->data->test->m, &nn->data->test->n);
        }
    }
    fprintf(stdout, "%s: done.\n", DEFAULT_CONSOLE_WRITER);
}
