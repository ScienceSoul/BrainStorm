//
//  Conv2DNetOps.h
//  BrainStorm
//
//  Created by Hakime Seddik on 13/08/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef Conv2DNetOps_h
#define Conv2DNetOps_h

void infer_convolution_op(void * _Nonnull  neural, unsigned int layer, unsigned int * _Nullable advance);
void infer_full_connected_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance);
void max_pooling_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance);
void l2_pooling_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance);
void average_pooling_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance);

void backpropag_full_connected_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance);
void backpropag_convolution_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance);

void backpropag_max_pooling_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance);
void backpropag_l2_pooling_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance);
void backpropag_average_pooling_op(void * _Nonnull neural, unsigned int layer, unsigned int * _Nullable advance);

void inference_in_conv2d_net(void * _Nonnull neural);
void backpropag_in_conv2d_net(void * _Nonnull neural,
                              void (* _Nullable ptr_inference_func)(void * _Nonnull self));
void batch_accumulation_in_conv2d_net(void * _Nonnull neural);

#endif /* Conv2DNetOps_h */
