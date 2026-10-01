EXE ?= erebus

# OpenBench builds with `make -j EXE=<path>`.
all:
	cargo build --release
	cp target/release/erebus $(EXE)

.PHONY: all
