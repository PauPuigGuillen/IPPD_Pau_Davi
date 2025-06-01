CC=nvcc
CFLAGS=-O2 -arch=sm_90
OBJ=vecadd_cuda

all:
	$(CC) $(OBJ).cu -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)
