//
//  Conv2DNetOps.h
//  BrainStorm
//
//  Created by Hakime Seddik on 13/08/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef Conv2DNetOps_h
#define Conv2DNetOps_h

void infer_convolution_op(void * _Nonnull  neural, unsigned int op, int * _Nullable advance);
void infer_fully_connected_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance);
void max_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance);
void l2_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance);
void average_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance);

void backpropag_full_connected_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance1, int * _Nullable advance2, int  * _Nullable advance3);
void backpropag_convolution_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance,  int * _Nullable advance2, int  * _Nullable advance3);

void backpropag_max_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance,  int * _Nullable advance2, int  * _Nullable advance3);
void backpropag_l2_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance, int * _Nullable advance2, int  * _Nullable advance3);
void backpropag_average_pooling_op(void * _Nonnull neural, unsigned int op, int * _Nullable advance, int * _Nullable advance2, int  * _Nullable advance3);

void inference_in_conv2d_net(void * _Nonnull neural);
void backpropag_in_conv2d_net(void * _Nonnull neural,
                              void (* _Nullable ptr_inference_func)(void * _Nonnull self));
void batch_accumulation_in_conv2d_net(void * _Nonnull neural);

void transpose_convolution(void * _Nonnull neural, unsigned int op,  int * _Nullable advance2);

#endif /* Conv2DNetOps_h */
