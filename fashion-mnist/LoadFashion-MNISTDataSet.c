//
//  LoadMNISTDataSet.c
//  mnist
//
//  Created by Hakime Seddik on 05/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#include <stdio.h>
#include "LoadFashion-MNISTDataSet.h"
#include "Utils.h"
#include "Memory.h"

void endianSwap(unsigned int *x) {
    *x = (*x>>24)|((*x<<8)&0x00FF0000)|((*x>>8)&0x0000FF00)|(*x<<24);
}

float * _Nullable * _Nullable readBinaryFile(const char * _Nonnull file, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2, unsigned int * _Nonnull num_channels, bool testData) {
    
    FILE *fimage = fopen(file, "rb");
    FILE *flabel = NULL;
    if (!fimage) {
        return NULL;
    } else {
        // We assume that the training set labels are located in the same directory as the training set images
        // So replace "train-images-idx3-ubyte" by "train-labels-idx1-ubyte" in the original path
        // Same assumption for the test set labels, replace t"10k-images-idx3-ubyte" by "t10k-labels-idx1-ubyte"
        // If we don't find the labels, we fatal...
        char *labelFile = malloc((strlen(file)*10)*sizeof(char));
        memset(labelFile, 0, strlen(file)*sizeof(char));
        char *word = NULL;
        if (!testData) {
            word = "train";
        } else {
            word = "t10k";
        }
        char *ptr = strstr(file, word);
        if (ptr != NULL) {
            if (!testData) {
                fprintf(stdout, "%s: file path to FASHION-MNIST training images file: %zu characters length.\n", DEFAULT_CONSOLE_WRITER, strlen(file));
            } else fprintf(stdout, "%s: file path to FASHION-MNIST test images file: %zu characters length.\n", DEFAULT_CONSOLE_WRITER, strlen(file));
            size_t firstCopyLength = strlen(file) - strlen(ptr);
            memcpy(labelFile, file, firstCopyLength*sizeof(char));
            if (!testData) {
                strcat(labelFile, "train-labels-idx1-ubyte");
            } else strcat(labelFile, "t10k-labels-idx1-ubyte");
        } else {
            if (!testData) {
                fatal(DEFAULT_CONSOLE_WRITER, "FASHION-MNIST training set images file not valid.");
            } else fatal(DEFAULT_CONSOLE_WRITER, "FASHION-MNIST test set images file not valid.");
        }
        flabel = fopen(labelFile, "rb");
        if (!flabel) {
            if (!testData) {
                fatal(DEFAULT_CONSOLE_WRITER, "Can't find the training set labels file \'train-labels-idx1-ubyte\'.");
            } else fatal(DEFAULT_CONSOLE_WRITER, "Can't find the test set labels file \'t10k-labels-idx1-ubyte\'.");
        }
        if (!testData) {
            fprintf(stdout, "%s: got the training set labels.\n", DEFAULT_CONSOLE_WRITER);
        } else fprintf(stdout, "%s: got the test set labels.\n", DEFAULT_CONSOLE_WRITER);
        free(labelFile);
    }
    
    unsigned int magic, num, row, col;
    // Check if magic numbers are valid
    fread(&magic, 4, 1, fimage);
    if (magic != 0x03080000) fatal(DEFAULT_CONSOLE_WRITER, "magic number in traning/test set images file not correct.");
    
    fread(&magic, 4, 1, flabel);
    if (magic != 0x01080000) fatal(DEFAULT_CONSOLE_WRITER, "magic number in traning/test set labels file not correct.");
    
    fread(&num, 4, 1, flabel); // Just advance in this file
    fread(&num, 4, 1, fimage); endianSwap(&num);
    fread(&row, 4, 1, fimage); endianSwap(&row);
    fread(&col, 4, 1, fimage); endianSwap(&col);
    
    if (!testData) {
        fprintf(stdout,"%s: number of examples in MNIST training set: %d\n", DEFAULT_CONSOLE_WRITER, num);
        fprintf(stdout,"%s: number of features in each MNIST example: %d x %d\n", DEFAULT_CONSOLE_WRITER, col, row);
    } else fprintf(stdout,"%s: number of examples in MNIST test set: %d\n", DEFAULT_CONSOLE_WRITER, num);
    
    fprintf(stdout, "---------------------------------------------------------------------\n");
    if (!testData) {
        fprintf(stdout, "Sample of the FASHION-MNIST training data set.\n");
    } else fprintf(stdout, "Sample of the FASHION-MNIST test data set.\n");
    fprintf(stdout, "---------------------------------------------------------------------\n");
    
    // Return a design matrix of the data set
    *len1 = num;
    *len2 = row*col + 1; // Number of features plus the ground-truth label
    *num_channels = 1;
    float **dataSet = floatmatrix(0, *len1-1, 0, *len2-1);
    memset(*dataSet, 0.0f, (*len1*(*len2))*sizeof(float));
    int idx;
    for (int ex=0; ex<num; ex++) {
        idx = 0;
        if (ex<10) fprintf(stdout,"---\n");
        for (int i=0; i<row; i++) {
            for (int j=0; j<col; j++) {
                unsigned char pixel;
                fread(&pixel, 1, 1, fimage);
                // Convert the pixel intensity value to a float
                dataSet[ex][idx] = (float)pixel;
                if (ex<10) {
                    // Just show a few examples (here 10) from the dataset to check if we got the data properly
                    // Output in hexadecimal
                    int byte = (int)dataSet[ex][idx];
                    fprintf(stdout,"%02x", byte);
                }
                // Normalize the intensity of each pixel from [0:255] to [0.0:1:0]
                dataSet[ex][idx] = dataSet[ex][idx] * (1.0f/255.0f);
                idx++;
            }
            if (ex<10) fprintf(stdout,"\n");
        }
        if (ex<10) fprintf(stdout,"\n");
        unsigned char label;
        fread(&label, 1, 1, flabel);
        dataSet[ex][idx] = (float)label;
        if (ex<10) fprintf(stdout,"label = %d\n", (int)dataSet[ex][idx]);
    }
    
    return dataSet;
}

float * _Nonnull * _Nonnull load_fashion_mnist(const char * _Nonnull file, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2, unsigned int * _Nonnull num_channels) {
    
    float **dataSet = readBinaryFile(file, len1, len2, num_channels, false);
    if (dataSet == NULL) {
        fatal(DEFAULT_CONSOLE_WRITER, "problem reading FASHION-MNIST data set.");
    } else fprintf(stdout, "%s: done.\n", DEFAULT_CONSOLE_WRITER);
    
    return dataSet;
}

float * _Nonnull * _Nonnull load_fashion_mnist_test(const char * _Nonnull file, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2, unsigned int * _Nonnull num_channels) {
    
    float **dataSet = readBinaryFile(file, len1, len2, num_channels, true);
    if (dataSet == NULL) {
        fatal(DEFAULT_CONSOLE_WRITER, "problem reading the FASHION-MNIST test data set ");
    } else fprintf(stdout, "%s: done.\n", DEFAULT_CONSOLE_WRITER);
    
    return dataSet;
}
