CC=nvc
CFLAGS=-O2 -lm -gpu=managed -acc=gpu -Minfo=all -mp
OBJ=partis_oacc_uni_mem

all:
	$(CC) $(OBJ).c -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)
