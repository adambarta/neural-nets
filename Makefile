CFLAGS += -DDEBUG
CFLAGS += -ggdb
CFLAGS += -O3
CFLAGS += -Wall

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
	$(CC) $(CFLAGS) -o $@ $^
