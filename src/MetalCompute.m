//
//  MetalCompute.m
//  FeedforwardNT
//
//  Created by Hakime Seddik on 08/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <sys/stat.h>

#include "NeuralNetwork.h"
#include "MetalCompute.h"
#include "Utils.h"
#include "TimeProfile.h"

id <MTLDevice> _Nullable device;
id <MTLCommandQueue> _Nullable commandQueue;
id <MTLLibrary> _Nullable library;

NSMutableArray *functions;
NSMutableArray *pipelineStates;

id <MTLBuffer> kernel_data;
id <MTLBuffer> kernel_weights;
id <MTLBuffer> kernel_biases;
id <MTLBuffer> kernel_activations;
id <MTLBuffer> kernel_ground_truth;
id <MTLBuffer> kernel_parameters;

bool allocation;

typedef struct parameters_container {
    unsigned int gridDimension;
    unsigned int numberOfLayers;
    unsigned int numberOfFeatures;
    unsigned int numberOfOutputs;
    
    weightMatrixDimension weightsDim[100];
    biasVectorDimension biasesDim[100];
} parameters_container;

int LoadFileIntoString(const char * _Nonnull filename, char * _Nonnull * _Nullable text, unsigned int * _Nonnull len) {
    struct stat statbuf;
    FILE        *fh;
    int         file_len;
    
    fh = fopen(filename, "r");
    if (fh == 0)
        return -1;
    
    stat(filename, &statbuf);
    file_len = (int)statbuf.st_size;
    *len = file_len;
    *text = (char *) malloc(file_len + 1);
    fread(*text, file_len, 1, fh);
    (*text)[file_len] = '\0';
    
    fclose(fh);
    return 0;
}

void init (void) {
    device = MTLCreateSystemDefaultDevice();
    commandQueue = device.newCommandQueue;
    
    char * metal_source;
    unsigned int src_len;
    
    if (LoadFileIntoString("metal/MetalKernels.metal", &metal_source, &src_len) != 0) {
        if (LoadFileIntoString("../metal/MetalKernels.metal", &metal_source, &src_len) != 0) {
            fatal(PROGRAM_NAME, "<metal compute>: can't load the metal source file.");
        }
    }
    
    NSError *error;
    library = [device newLibraryWithSource:[NSString stringWithUTF8String:metal_source] options:NULL error:&error];
    if (error != nil) {
        fprintf(stderr, "<metal compute>: error when creating a new library state.");
        fprintf(stderr, "<metal compute>: error code: %ld\n", (long)error.code);
        fatal(PROGRAM_NAME, "Program will abort.");
    }
    
    functions = [NSMutableArray new];
    pipelineStates = [NSMutableArray new];
    
    allocation = false;
}

void nullify(void) {
    device = nil;
    commandQueue = nil;
    library = nil;
    functions = nil;
    pipelineStates = nil;
    
    kernel_data = nil;
    kernel_weights = nil;
    kernel_biases = nil;
    kernel_activations = nil;
    kernel_parameters  = nil;
    kernel_ground_truth = nil;
}

void allocate_buffers(void * _Nonnull network) {
    if (!allocation) {
        
        NeuralNetwork *nn = (NeuralNetwork *)network;
        parameters_container *params = (parameters_container *)malloc(sizeof(parameters_container));
        
        unsigned int entriesTableSize = nn->data->test->m * nn->number_of_features;
        
        unsigned int weightsTableSize = 0;
        unsigned int biasesTableSize = 0;
        for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
            weightsTableSize = weightsTableSize + (nn->weightsDimensions[l].m*nn->weightsDimensions[l].n);
            biasesTableSize = biasesTableSize + nn->biasesDimensions[l].n;
        }
        
        int max = max_array(nn->parameters->topology, nn->parameters->numberOfLayers);
        unsigned int activationsTableSize = max * nn->data->test->m;
        
        kernel_data = [device newBufferWithLength:entriesTableSize*sizeof(float) options:MTLResourceStorageModeShared];
        kernel_weights = [device newBufferWithLength:weightsTableSize*sizeof(float) options:MTLResourceStorageModeShared];
        kernel_biases = [device newBufferWithLength:biasesTableSize*sizeof(float) options:MTLResourceStorageModeShared];
        kernel_activations = [device newBufferWithLength:activationsTableSize*sizeof(float) options:MTLResourceStorageModePrivate];
        kernel_ground_truth = [device newBufferWithLength:nn->data->test->m*sizeof(float) options:MTLResourceStorageModeShared];
        
        params->gridDimension = nn->data->test->m;
        params->numberOfLayers = nn->parameters->numberOfLayers;
        params->numberOfFeatures = nn->parameters->topology[0];
        params->numberOfOutputs = nn->parameters->topology[nn->parameters->numberOfLayers-1];
        memcpy(params->weightsDim, nn->weightsDimensions, sizeof(nn->weightsDimensions));
        memcpy(params->biasesDim, nn->biasesDimensions, sizeof(nn->biasesDimensions));
        kernel_parameters = [device newBufferWithBytes:params length:sizeof(parameters_container) options:MTLResourceStorageModeShared];
        free(params);
        
        void *buffer = kernel_data.contents;
        memset(buffer, 0.0f, entriesTableSize*sizeof(float));
        
        buffer = kernel_weights.contents;
        memset(buffer, 0.0f, weightsTableSize*sizeof(float));
        
        buffer = kernel_biases.contents;
        memset(buffer, 0.0f, biasesTableSize*sizeof(float));
    }
}

void prepare (char * _Nonnull operation) {
    
    id <MTLFunction> function;
    id <MTLComputePipelineState> pipelineState;
    
    if (!allocation) {
        if (strcmp(operation, "feedforward") == 0) {
            function = [library newFunctionWithName:[NSString stringWithUTF8String:"feedforward"]];
            [functions addObject:function];
            
            NSError *error;
            pipelineState = [device newComputePipelineStateWithFunction:functions[0] error:&error];
            if (error != nil) {
                fprintf(stderr, "<metal compute>: error when creating a pipeline state.");
                fprintf(stderr, "<metal compute>: error code: %ld\n", (long)error.code);
                fatal(PROGRAM_NAME, "Program will abort.");
            }
            
            [functions addObject:function];
            [pipelineStates addObject:pipelineState];
        }
        allocation = true;
    }
}

void format_data(float * _Nonnull * _Nonnull input, unsigned int m, unsigned int n) {
    
    static bool firstTime = false;
    
    if (firstTime) return;
    
    if (!firstTime) {
        float *mat = (float *)malloc((m*n)*sizeof(float));
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                mat[(m*j)+i] = input[i][j];
            }
        }
        
        void *buffer = kernel_data.contents;
        memcpy(buffer, mat, (m*n)*sizeof(float));
        
        free(mat);
        firstTime = true;
    }
}

void compute_feedforward(void * _Nonnull neural, float * _Nonnull result) {
    
    MTLSize threadgroupsPerGrid;
    MTLSize threadsPerThreadgroup;
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    unsigned int weightsTableSize = 0;
    unsigned int biasesTableSize = 0;
    for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
        weightsTableSize = weightsTableSize + (nn->weightsDimensions[l].m * nn->weightsDimensions[l].n);
        biasesTableSize = biasesTableSize + nn->biasesDimensions[l].n;
    }
    
    void *buffer = kernel_weights.contents;
    memcpy(buffer, nn->weights, weightsTableSize*sizeof(float));
    
    buffer = kernel_biases.contents;
    memcpy(buffer, nn->biases, biasesTableSize*sizeof(float));
    
    buffer = kernel_ground_truth.contents;
    float *pt = buffer;
    for (int i=0; i<nn->data->test->m; i++) {
        pt[i] = nn->data->test->set[i][nn->number_of_features];
    }
    
    @autoreleasepool{
        
        id <MTLComputePipelineState> pipelineState = pipelineStates[0];
        unsigned long threadExecutionWidth = pipelineState.threadExecutionWidth;
        
        threadgroupsPerGrid = MTLSizeMake((nn->data->test->m + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
        threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
        
        id <MTLCommandBuffer> commandBuffer = commandQueue.commandBuffer;
        id <MTLComputeCommandEncoder> commandEncoder = commandBuffer.computeCommandEncoder;
        [commandEncoder setComputePipelineState:pipelineStates[0]];
        
        [commandEncoder setBuffer:kernel_data offset:0 atIndex:0];
        [commandEncoder setBuffer:kernel_weights offset:0 atIndex:1];
        [commandEncoder setBuffer:kernel_biases offset:0 atIndex:2];
        [commandEncoder setBuffer:kernel_activations offset:0 atIndex:3];
        [commandEncoder setBuffer:kernel_ground_truth offset:0 atIndex:4];
        [commandEncoder setBuffer:kernel_parameters offset:0 atIndex:5];
        
        [commandEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    void *output = kernel_ground_truth.contents;
    memcpy(result, output, nn->data->test->m*sizeof(float));
}

MetalCompute * _Nonnull metalCompute(void) {
    
    MetalCompute *metalComppute = (MetalCompute *)malloc(sizeof(MetalCompute));
    
    metalComppute->init = init;
    metalComppute->prepare = prepare;
    metalComppute->allocate_buffers = allocate_buffers;
    metalComppute->nullify = nullify;
    
    metalComppute->format_data = format_data;
    metalComppute->feedforward = compute_feedforward;
    
    return metalComppute;
}

