//
//  TimeProfile.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 28/05/2017.
//

#include <sys/types.h>
#include <sys/times.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <time.h>

#include "TimeProfile.h"

static struct rusage usage;

static struct timeval tp;
static struct timezone tzp;

double cputime (void) {
    
    getrusage( RUSAGE_SELF, &usage );
    return (double) usage.ru_utime.tv_sec + usage.ru_utime.tv_usec*1.0e-6;
}

double realtime (void) {
    
    gettimeofday( &tp,&tzp );
    return (double) tp.tv_sec + tp.tv_usec*1.0e-6;
}

double cpumemory (void) {
    
    getrusage( RUSAGE_SELF, &usage );
    return (double) 1.0 * usage.ru_maxrss;
}
