//
//  DenseNetOps.h
//  BrainStorm
//
//  Created by Hakime Seddik on 13/08/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef DenseNetOps_h
#define DenseNetOps_h

void inference_in_dense_net(void * _Nonnull neural);
void backpropag_in_dense_net(void * _Nonnull neural,
                             void (* _Nullable ptr_inference_func)(void * _Nonnull self));
void batch_accumulation_in_dense_net(void * _Nonnull neural);

#endif /* DenseNetOps_h */
