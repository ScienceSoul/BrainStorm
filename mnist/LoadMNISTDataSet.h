//
//  LoadMNISTDataSet.h
//  mnist
//
//  Created by Hakime Seddik on 05/07/2018.
//  Copyright © 2018 Hakime Seddik. All rights reserved.
//

#ifndef LoadMNISTDataSet_h
#define LoadMNISTDataSet_h

float * _Nonnull * _Nonnull loadMnist(const char * _Nonnull file, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2);
float * _Nonnull * _Nonnull loadMnistTest(const char * _Nonnull file, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2);

#endif /* LoadMNISTDataSet_h */
