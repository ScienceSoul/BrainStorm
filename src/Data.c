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

static void * _Nonnull set_data(void * _Nonnull self, float * _Nonnull * _Nonnull dataSet, unsigned int start, unsigned int end, int * _Nullable dense_topo, int conv_topo[_Nullable][9], unsigned int num_channels) {
    
    brain_storm_net *nn = (brain_storm_net *)self;
    
    tensor_dict *dict = init_tensor_dict();
    
    if (dense_topo != NULL) {
        dict->rank = 1;
        dict->shape[0][0][0] = end * dense_topo[0];
    } else if (conv_topo != NULL) {
        dict->rank = 4;
        dict->shape[0][0][0] = end;
        dict->shape[0][1][0] = conv_topo[0][2];
        dict->shape[0][2][0] = conv_topo[0][3];
        dict->shape[0][3][0] = num_channels;
    } else {
        fatal(DEFAULT_CONSOLE_WRITER, "topology missing for data creation.");
    }
    
   void *set = nn->tensor(NULL, *dict);
    if (set == NULL) {
        fatal(DEFAULT_CONSOLE_WRITER, "training data set allocation error.");
    }
    
    int fh = 0;
    int fw = 0;
    if (nn->is_dense_network) {
        
        fh = dense_topo[0];
        tensor *t = (tensor *)set;
        
        int indx = 0;
        for (int l=(int)start; l<start+end; l++) {
            for (int i=0; i<fh; i++) {
                t->val[indx] = dataSet[l][i];
                indx++;
            }
        }
    } else if (nn->is_conv2d_network) {
        
        fh = conv_topo[0][2];
        fw = conv_topo[0][3];
        
        tensor *t = (tensor *)set;
        
        int stride1 = 0;
        for (int l=(int)start; l<start+end; l++) {
            int indx = 0;
            int stride2 = 0;
            for (int i=0; i<fh; i++) {
                for (int j=0; j<fw; j++) {
                    for (int k=0; k<num_channels; k++) {
                        t->val[stride1+(stride2+(j*num_channels+k))] = dataSet[l][indx];
                        indx++;
                    }
                }
                stride2 = stride2 + (fw * num_channels);
            }
            stride1 = stride1 + (fh * fw * num_channels);
        }
    }
    
    free(dict);
    
    return set;
}

static void * _Nonnull set_labels(void * _Nonnull self, float * _Nonnull * _Nonnull dataSet, unsigned int start, unsigned int end, int * _Nullable classifications, unsigned int numberOfClassifications, int * _Nullable dense_topo, int conv_topo[_Nullable][9], unsigned int num_channels, unsigned int numberOfLayers, bool binarization) {
    
    brain_storm_net *nn = (brain_storm_net *)self;
    
    if (numberOfClassifications > 0) {
        
        if (classifications == NULL) {
            fatal(DEFAULT_CONSOLE_WRITER, "classifications array is NULL.");
        }
        
        bool classsification_error = false;
        if (nn->is_dense_network) {
            if (dense_topo[numberOfLayers-1] != numberOfClassifications) classsification_error = true;
        } else if (nn->is_conv2d_network) {
            if (conv_topo[numberOfLayers-1][1] != numberOfClassifications) classsification_error = true;
        }
        if (classsification_error) {
            fatal(DEFAULT_CONSOLE_WRITER, "the number of classifications should be equal to the number of activations at the output layer.");
        }
    }
    
    int label_indx = 0;
    if (nn->is_dense_network) {
        label_indx = dense_topo[0];
    } else if (nn->is_conv2d_network) {
        label_indx = conv_topo[0][2] * conv_topo[0][3] * num_channels;
    }
    
    tensor_dict *dict = init_tensor_dict();
    
    void *labels;
    if (numberOfClassifications > 0) {
        dict->rank = 1;
        dict->shape[0][0][0] = end * numberOfClassifications;
        labels = nn->tensor(NULL, *dict);
        tensor *t = (tensor *)labels;
        if (binarization) {
            int indx = 0;
            for (int l=(int)start; l<start+end; l++) {
                for (int i=0; i<numberOfClassifications; i++) {
                    if (dataSet[l][label_indx] == (float)classifications[i]) {
                        t->val[indx] = 1.0f;
                    } else {
                        t->val[indx] = 0.0;
                    }
                    indx++;
                }
            }
        } else {
            int indx = 0;
            for (int l=(int)start; l<start+end; l++) {
                for (int i=0; i<numberOfClassifications; i++) {
                    if (dataSet[l][label_indx] == (float)classifications[i]) {
                        t->val[indx] = dataSet[l][label_indx];
                    }
                    indx++;
                }
            }
        }
    } else {
        dict->rank = 1;
        dict->shape[0][0][0] = end;
        labels = nn->tensor(NULL, *dict);
        tensor *t = (tensor *)labels;
        int indx = 0;
        for (int l=(int)start; l<start+end; l++) {
            t->val[indx] = dataSet[l][label_indx];
            indx++;
        }
    }
    
    free(dict);
    
    return labels;
}

void load_data(void * _Nonnull self, const char * _Nonnull dataSetName, const char * _Nonnull trainFile, const char * _Nonnull testFile, bool testData, bool binarization) {
    
    unsigned int len1=0, len2=0, num_channels;
    float **raw = NULL;
    
    brain_storm_net *nn = (brain_storm_net *)self;
    
    fprintf(stdout, "%s: load the <%s> training data set...\n", DEFAULT_CONSOLE_WRITER, dataSetName);
    raw = nn->data->training->reader(trainFile, &len1, &len2, &num_channels);
    shuffle(raw, len1, len2);
    
    if (nn->is_dense_network) {
        
        nn->data->training->set = set_data(self, raw, 0, nn->dense->parameters->split[0], nn->dense->parameters->topology, NULL, num_channels);
        nn->data->training->labels = set_labels(self, raw, 0, nn->dense->parameters->split[0], nn->dense->parameters->classifications, nn->dense->parameters->number_of_classifications, nn->dense->parameters->topology, NULL, num_channels, nn->network_num_layers, binarization);
        
        
    } else if (nn->is_conv2d_network) {
        
        nn->data->training->set = set_data(self, raw, 0, nn->conv2d->parameters->split[0], NULL, nn->conv2d->parameters->topology, num_channels);
        nn->data->training->labels = set_labels(self, raw, 0, nn->conv2d->parameters->split[0], nn->conv2d->parameters->classifications, nn->conv2d->parameters->number_of_classifications, NULL, nn->conv2d->parameters->topology, num_channels, nn->network_num_layers, binarization);
    }
    
    if (testData) {
        fprintf(stdout, "%s: load test data set in <%s>...\n", DEFAULT_CONSOLE_WRITER, dataSetName);
        
        unsigned int len1_test=0, len2_test=0;
        float **raw_test = nn->data->test->reader(testFile, &len1_test, &len2_test, &num_channels);
        
        if (nn->is_dense_network) {
            nn->data->test->set = set_data(self, raw_test, 0, len1_test, nn->dense->parameters->topology, NULL, num_channels);
            nn->data->test->labels = set_labels(self, raw_test, 0, len1_test, NULL, 0, nn->dense->parameters->topology, NULL, num_channels, nn->network_num_layers, false);
            
            nn->data->validation->set = set_data(self, raw, nn->dense->parameters->split[0], nn->dense->parameters->split[1], nn->dense->parameters->topology, NULL, num_channels);
            nn->data->validation->labels = set_labels(self, raw, nn->dense->parameters->split[0], nn->dense->parameters->split[1], NULL, 0, nn->dense->parameters->topology, NULL, num_channels, nn->network_num_layers, false);
        } else if (nn->is_conv2d_network) {
            nn->data->test->set = set_data(self, raw_test, 0, len1_test, NULL, nn->conv2d->parameters->topology, num_channels);
            nn->data->test->labels = set_labels(self, raw_test, 0, len1_test, NULL, 0, NULL, nn->conv2d->parameters->topology, num_channels, nn->network_num_layers, false);
            
            nn->data->validation->set = set_data(self, raw, nn->conv2d->parameters->split[0], nn->conv2d->parameters->split[1], NULL, nn->conv2d->parameters->topology, num_channels);
            nn->data->validation->labels = set_labels(self, raw, nn->conv2d->parameters->split[0], nn->conv2d->parameters->split[1], NULL, 0, NULL, nn->conv2d->parameters->topology, num_channels, nn->network_num_layers, false);
        }
    } else {
        if (nn->is_dense_network) {
            nn->data->test->set = set_data(self, raw,  nn->dense->parameters->split[0],  nn->dense->parameters->split[1], nn->dense->parameters->topology, NULL, num_channels);
            nn->data->test->labels = set_labels(self, raw,  nn->dense->parameters->split[0],  nn->dense->parameters->split[1], NULL, 0, nn->dense->parameters->topology, NULL, num_channels, nn->network_num_layers, false);
        } else if (nn->is_conv2d_network) {
            nn->data->test->set = set_data(self, raw,  nn->conv2d->parameters->split[0],  nn->conv2d->parameters->split[1], NULL, nn->conv2d->parameters->topology, num_channels);
            nn->data->test->labels = set_labels(self, raw,  nn->conv2d->parameters->split[0],  nn->conv2d->parameters->split[1], NULL, 0, NULL, nn->conv2d->parameters->topology, num_channels, nn->network_num_layers, false);
        }
    }
    fprintf(stdout, "%s: load done.\n", DEFAULT_CONSOLE_WRITER);
}
