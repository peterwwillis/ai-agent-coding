.PHONY: all build vet test clean

# Extract the authoritative version from the source file.
VERSION := $(shell grep 'var version = ' main.go | sed 's/.*"\(.*\)".*/\1/')
LDFLAGS  := -ldflags "-X main.version=$(VERSION)"

all: build

## build: compile the sgpt binary, embedding the version from main.go
build:
	go build $(LDFLAGS) -o sgpt .

## vet: run go vet on all packages
vet:
	go vet ./...

## test: run all unit tests
test:
	go test ./... -v

## clean: remove the compiled binary
clean:
	rm -f sgpt
