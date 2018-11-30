
#
# 'make depend' uses makedepend to automatically generate dependencies
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'FeedforwardNT'
# 'make clean'  removes all .o and executable files
#

PLATEFORM = $(shell uname -s)

ifeq ($(PLATEFORM),Darwin)
	CC = clang
else
	CC = clang-7
endif

# Compile flags
CFLAGS = -Wall -Ofast

ifeq ($(PLATEFORM),Darwin)
	INCLUDES = -I./ -I../BrainStorm/include 
else
	INCLUDES = -I./ -I../BrainStorm/include -I/usr/local/include 
endif

ifeq ($(PLATEFORM),Darwin)
	LFLAGS = -L../BrainStorm/src -lBrainStorm -framework Accelerate
else
	LFLAGS = -L../BrainStorm/src -lBrainStorm -L/usr/local/lib -lopenblas 
endif

ifeq ($(PLATEFORM),Darwin)
	LIBS = -lm
else
	LIBS = -lm -lbsd -lgfortran
endif

SRCS = $(shell echo *.c)

# define the C object files
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.c=.o)

# define the executable file
MAIN = fashion-mnist

.PHONY: depend clean

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN) Makefile.bak

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
