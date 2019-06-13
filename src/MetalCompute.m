//
//  MetalCompute.m
//  FeedforwardNT
//
//  Created by Hakime Seddik on 08/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifdef GPU_WORKING

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <sys/stat.h>

#include "NeuralNetwork.h"
#include "MetalCompute.h"
#include "Utils.h"
#include "TimeProfile.h"

id <MTLDevice> _Nullable device;
id <MTLCommandQueue> _Nullable command_queue;
id <MTLLibrary> _Nullable library;

NSMutableArray *functions;
NSMutableArray *pipeline_states;

id <MTLBuffer> kernel_data;
id <MTLBuffer> kernel_weights;
id <MTLBuffer> kernel_biases;
id <MTLBuffer> kernel_activations;
id <MTLBuffer> kernel_ground_truth;
id <MTLBuffer> kernel_parameters;

bool allocation;

typedef struct weight_matrix_dimension {
    unsigned int m, n;
} weight_matrix_dimension;

typedef struct bias_vector_dimension {
    unsigned int n;
} bias_vector_dimension;

typedef struct parameters_container {
    unsigned int grid_dim;
    unsigned int num_layers;
    unsigned int num_features;
    unsigned int num_outputs;
    
    weight_matrix_dimension weights_dim[100];
    bias_vector_dimension biases_dim[100];
} parameters_container;

int LoadFileIntoString(const char * _Nonnull file_name, char * _Nonnull * _Nullable text, unsigned int * _Nonnull len) {
    struct stat statbuf;
    FILE        *fh;
    int         file_len;
    
    fh = fopen(file_name, "r");
    if (fh == 0)
        return -1;
    
    stat(file_name, &statbuf);
    file_len = (int)statbuf.st_size;
    *len = file_len;
    *text = (char *) malloc(file_len + 1);
    fread(*text, file_len, 1, fh);
    (*text)[file_len] = '\0';
    
    fclose(fh);
    return 0;
}

void init_device (void) {
    device = MTLCreateSystemDefaultDevice();
    command_queue = device.newCommandQueue;
    
    char * metal_source;
    unsigned int src_len;
    
    if (LoadFileIntoString("metal/MetalKernels.metal", &metal_source, &src_len) != 0) {
        if (LoadFileIntoString("../metal/MetalKernels.metal", &metal_source, &src_len) != 0) {
            fatal(DEFAULT_CONSOLE_WRITER, "<metal compute>: can't load the metal source file.");
        }
    }
    
    NSError *error;
    library = [device newLibraryWithSource:[NSString stringWithUTF8String:metal_source] options:NULL error:&error];
    if (error != nil) {
        fprintf(stderr, "<metal compute>: error when creating a new library state.");
        fprintf(stderr, "<metal compute>: error code: %ld\n", (long)error.code);
        fatal(DEFAULT_CONSOLE_WRITER, "Program will abort.");
    }
    
    functions = [NSMutableArray new];
    pipeline_states = [NSMutableArray new];
    
    allocation = false;
}

void nullify(void) {
    device = nil;
    command_queue = nil;
    library = nil;
    functions = nil;
    pipeline_states = nil;
    
    kernel_data = nil;
    kernel_weights = nil;
    kernel_biases = nil;
    kernel_activations = nil;
    kernel_parameters  = nil;
    kernel_ground_truth = nil;
}

void allocate_buffers(void * _Nonnull network) {
    if (!allocation) {
        
        brain_storm_net *nn = (brain_storm_net *)network;
        parameters_container *params = (parameters_container *)malloc(sizeof(parameters_container));
        
        unsigned int entries_table_size = nn->data->test->m * nn->num_channels;
        
        unsigned int weights_table_size = 0;
        unsigned int biases_table_size = 0;
        for (int l=0; l<nn->network_num_layers-1; l++) {
            weights_table_size = weights_table_size + (nn->dense->weights->shape[l][0][0]*nn->dense->weights->shape[l][1][0]);
            biases_table_size = biases_table_size + nn->dense->biases->shape[l][0][0];
        }
        
        int max = max_array(nn->dense->parameters->topology, nn->network_num_layers);
        unsigned int activations_table_size = max * nn->data->test->m;
        
        kernel_data = [device newBufferWithLength:entries_table_size*sizeof(float) options:MTLResourceStorageModeShared];
        kernel_weights = [device newBufferWithLength:weights_table_size*sizeof(float) options:MTLResourceStorageModeShared];
        kernel_biases = [device newBufferWithLength:biases_table_size*sizeof(float) options:MTLResourceStorageModeShared];
        kernel_activations = [device newBufferWithLength:activations_table_size*sizeof(float) options:MTLResourceStorageModePrivate];
        kernel_ground_truth = [device newBufferWithLength:nn->data->test->m*sizeof(float) options:MTLResourceStorageModeShared];
        
        params->grid_dim = nn->data->test->m;
        params->num_layers = nn->network_num_layers;
        params->num_features = nn->dense->parameters->topology[0];
        params->num_outputs = nn->dense->parameters->topology[nn->network_num_layers-1];
        
        for (int l=0; l<nn->network_num_layers-1; l++) {
            params->weights_dim[l].m = nn->dense->weights->shape[l][0][0];
            params->weights_dim[l].n = nn->dense->weights->shape[l][1][0];
            
            params->biases_dim[l].n = nn->dense->biases->shape[l][0][0];
        }

        kernel_parameters = [device newBufferWithBytes:params length:sizeof(parameters_container) options:MTLResourceStorageModeShared];
        free(params);
        
        void *buffer = kernel_data.contents;
        memset(buffer, 0.0f, entries_table_size*sizeof(float));
        
        buffer = kernel_weights.contents;
        memset(buffer, 0.0f, weights_table_size*sizeof(float));
        
        buffer = kernel_biases.contents;
        memset(buffer, 0.0f, biases_table_size*sizeof(float));
    }
}

void prepare (char * _Nonnull operation) {
    
    id <MTLFunction> function;
    id <MTLComputePipelineState> pipeline_state;
    
    if (!allocation) {
        if (strcmp(operation, "feedforward") == 0) {
            function = [library newFunctionWithName:[NSString stringWithUTF8String:"feedforward"]];
            [functions addObject:function];
            
            NSError *error;
            pipeline_state = [device newComputePipelineStateWithFunction:functions[0] error:&error];
            if (error != nil) {
                fprintf(stderr, "<metal compute>: error when creating a pipeline state.");
                fprintf(stderr, "<metal compute>: error code: %ld\n", (long)error.code);
                fatal(DEFAULT_CONSOLE_WRITER, "Program will abort.");
            }
            
            [functions addObject:function];
            [pipeline_states addObject:pipeline_state];
        }
        allocation = true;
    }
}

void format_data(float * _Nonnull * _Nonnull input, unsigned int m, unsigned int n) {
    
    static bool first_time = false;
    
    if (first_time) return;
    
    if (!first_time) {
        float *mat = (float *)malloc((m*n)*sizeof(float));
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                mat[(m*j)+i] = input[i][j];
            }
        }
        
        void *buffer = kernel_data.contents;
        memcpy(buffer, mat, (m*n)*sizeof(float));
        
        free(mat);
        first_time = true;
    }
}

void compute_feedforward(void * _Nonnull neural, float * _Nonnull result) {
    
    MTLSize thread_groups_per_grid;
    MTLSize threads_per_thread_group;
    
    brain_storm_net *nn = (brain_storm_net *)neural;
    
    unsigned int weights_table_size = 0;
    unsigned int biases_table_size = 0;
    for (int l=0; l<nn->network_num_layers-1; l++) {
        weights_table_size = weights_table_size + (nn->dense->weights->shape[l][0][0]*nn->dense->weights->shape[l][1][0]);
        biases_table_size = biases_table_size + nn->dense->biases->shape[l][0][0];
    }
    
    void *buffer = kernel_weights.contents;
    memcpy(buffer, nn->dense->weights, weights_table_size*sizeof(float));
    
    buffer = kernel_biases.contents;
    memcpy(buffer, nn->dense->biases, biases_table_size*sizeof(float));
    
    buffer = kernel_ground_truth.contents;
    float *pt = buffer;
    for (int i=0; i<nn->data->test->m; i++) {
        pt[i] = nn->data->test->set[i][nn->num_channels];
    }
    
    @autoreleasepool{
        
        id <MTLComputePipelineState> pipeline_state = pipeline_states[0];
        unsigned long thread_execution_width = pipeline_state.threadExecutionWidth;
        
        thread_groups_per_grid = MTLSizeMake((nn->data->test->m + thread_execution_width - 1) / thread_execution_width, 1, 1);
        threads_per_thread_group = MTLSizeMake(thread_execution_width, 1, 1);
        
        id <MTLCommandBuffer> command_buffer = command_queue.commandBuffer;
        id <MTLComputeCommandEncoder> command_encoder = command_buffer.computeCommandEncoder;
        [command_encoder setComputePipelineState:pipeline_states[0]];
        
        [command_encoder setBuffer:kernel_data offset:0 atIndex:0];
        [command_encoder setBuffer:kernel_weights offset:0 atIndex:1];
        [command_encoder setBuffer:kernel_biases offset:0 atIndex:2];
        [command_encoder setBuffer:kernel_activations offset:0 atIndex:3];
        [command_encoder setBuffer:kernel_ground_truth offset:0 atIndex:4];
        [command_encoder setBuffer:kernel_parameters offset:0 atIndex:5];
        
        [command_encoder dispatchThreadgroups:thread_groups_per_grid threadsPerThreadgroup:threads_per_thread_group];
        
        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
    
    void *output = kernel_ground_truth.contents;
    memcpy(result, output, nn->data->test->m*sizeof(float));
}

metal_compute * _Nonnull metal_compute_alloc(void) {
    
    metal_compute *new_compute = (metal_compute *)malloc(sizeof(metal_compute));
    
    new_compute->init = init_device;
    new_compute->prepare = prepare;
    new_compute->allocate_buffers = allocate_buffers;
    new_compute->nullify = nullify;
    
    new_compute->format_data = format_data;
    new_compute->feedforward = compute_feedforward;
    
    return new_compute;
}

#endif


