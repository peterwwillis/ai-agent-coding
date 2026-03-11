.PHONY: all build vet test clean

all: build

## build: compile the sgpt binary
build:
	go build -o sgpt .

## vet: run go vet on all packages
vet:
	go vet ./...

## test: run all unit tests
test:
	go test ./... -v

## clean: remove the compiled binary
clean:
	rm -f sgpt
