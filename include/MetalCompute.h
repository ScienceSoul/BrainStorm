//
//  MetalCompute.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 08/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef MetalCompute_h
#define MetalCompute_h


typedef struct MetalCompute {
    
    void (* _Nullable init)(void);
    void (* _Nullable prepare)(char * _Nonnull operation);
    void (* _Nullable allocate_buffers)(void * _Nonnull network);
    void (* _Nullable nullify)(void);
    
    void (* _Nullable format_data)(float * _Nonnull * _Nonnull input, unsigned int m, unsigned int n);
    void (* _Nullable feedforward)(void * _Nonnull neural, float * _Nonnull result);
    
} MetalCompute;

MetalCompute * _Nonnull metalCompute(void);

#endif /* MetalCompute_h */
