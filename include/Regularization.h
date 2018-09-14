//
//  Regularization.h
//  BrainStorm
//
//  Created by Hakime Seddik on 12/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef Regularization_h
#define Regularization_h

float l0_regularizer(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int stride, int stride1, int stride2);
float l1_regularizer(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int stride, int stride1, int stride2);
float l2_regularizer(void * _Nonnull neural, float * _Nonnull weights, float eta, float lambda, int i, int j, int n, int stride, int stride1, int stride2);

#endif /* Regularization_h */
