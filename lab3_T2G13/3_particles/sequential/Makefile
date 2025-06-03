CC=gcc
CFLAGS=-O2 -lm #-arch=sm_90 -acc=gpu -gpu=managed -Minfo=all -mp
OBJ=partis_seq

all:
	$(CC) $(OBJ).c -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)
