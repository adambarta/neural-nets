CFLAGS += -DDEBUG
CFLAGS += -ggdb
CFLAGS += -O3
CFLAGS += -Wall
LIBS = -lm
LDIR = -L.

PWD = $(shell pwd)

NNLIB = libnn.so

EXE = nn
EXESRC = eg.c

SRC = nn.c tf.c
HDR = nn.h

OBJ = $(patsubst %.c,%.o, $(SRC))

all: $(EXE) $(NNLIB)

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@ -fPIC


$(NNLIB): $(OBJ)
	$(CC) -o $@ $^ -shared -Wl,-x $(LIBS)


$(EXE): $(EXESRC) $(NNLIB)
	$(CC) $< -o $@ -lnn $(LDIR)

clean:
	rm -rf core *.o $(EXE) $(NNLIB)

install:
	ln -s $(PWD)/$(NNLIB) $(PREFIX)/lib/$(NNLIB)

