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
ifeq ($(PLATEFORM),Darwin)
	CFLAGS = -Wall -fPIC -Ofast -flto
else
	CFLAGS = -Wall -fPIC -Ofast
endif

ifeq ($(PLATEFORM),Darwin)
	INCLUDES = -I../include
else
	INCLUDES = -I../include -I/usr/local/include
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
STATIC = libBrainStorm.a

.PHONY: depend clean

$(STATIC): $(OBJS)
	@echo "[Link (Static)]"
	ar rcs $@ $^
	ranlib $(STATIC)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

all: $(STATIC)

clean:
	rm -f $(OBJS) $(STATIC) Makefile.bak

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
