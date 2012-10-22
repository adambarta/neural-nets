CFLAGS += -DDEBUG
CFLAGS += -ggdb
CFLAGS += -O3
CFLAGS += -Wall
LIBS = -lm

NNLIB = libnn.so
EXE = nn

SRC = nn.c
HDR = nn.h

OBJ = $(patsubst %.c,%.o, $(SRC))

all: $(EXE)

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(NNLIB): $(SRC) $(HDR)
	$(CC) -o $@ $^ -shared -Wl,-x

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(LIBS)

clean:
	rm -rf core *.o $(EXE)
