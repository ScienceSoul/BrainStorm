//
//  Conv2DNetOps.c
//  BrainStorm
//
//  Created by Hakime Seddik on 13/08/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include "cblas.h"
#endif

#include "NeuralNetwork.h"
#include "Conv2DNetOps.h"

void inference_in_conv2d_net(void * _Nonnull neural) {
    
}

void backpropag_in_conv2d_net(void * _Nonnull neural,
                              void (* _Nullable ptr_inference_func)(void * _Nonnull self)) {
    
}

void batch_accumulation_in_conv2d_net(void * _Nonnull neural) {
    
}
