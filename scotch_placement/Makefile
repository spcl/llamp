CC=gcc
CFLAGS=-g -O3 -Wno-deprecated -Wall
LDFLAGS=-lscotch -lscotcherr -lm

%.o: %.c *h
	$(CC) $(CCFLAGS) -c $<

all: calc_placement.o argparse.o
	$(CC) $(CLAGS) *.o -o scotch_placement $(LDFLAGS)

clean:
	rm -f *.o
	rm -f scotch_placement
