CC=nvc
CFLAGS=-O2 -lm -gpu=cc90 -acc=gpu -Minfo=all -mp
OBJ=partis_oacc_async

all:
	$(CC) $(OBJ).c -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)
