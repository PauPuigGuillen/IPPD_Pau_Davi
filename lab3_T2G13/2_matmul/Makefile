CC=nvcc
CFLAGS=-O2 -Xcompiler -Wall -std=c++11 -arch=sm_90 -lcublas
OBJ=matmul

all:
	$(CC) $(CFLAGS) $(OBJ).cu -o $(OBJ) $(LDFLAGS)

clean:
	rm $(OBJ)
