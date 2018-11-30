//
//  LoadMNISTDataSet.h
//  mnist
//
//  Created by Hakime Seddik on 05/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef LoadMNISTDataSet_h
#define LoadMNISTDataSet_h

float * _Nonnull * _Nonnull load_mnist(const char * _Nonnull file, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2, unsigned int * _Nonnull num_channels);
float * _Nonnull * _Nonnull load_mnist_test(const char * _Nonnull file, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2, unsigned int * _Nonnull num_channels);

#endif /* LoadMNISTDataSet_h */
