all: build

build:
	go build -o gon

test:
	go test ./neuralnet

run:
	./run.sh

clean:
	rm -f gon

.PHONY: all build test run clean
