CC=nvc
CFLAGS=-O2 #-arch=sm_90 -acc=gpu -gpu=managed -Minfo=all -mp
OBJ=vecadd_oacc

all:
	$(CC) $(OBJ).c -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)
