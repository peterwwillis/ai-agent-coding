
GIT_ROOTDIR := $(shell git rev-parse --show-toplevel)

install:
	command -v llama-swap || \
        $(GIT_ROOTDIR)/bin/llama-swap-install.sh

install-config:
	mkdir -p ~/.config/llama-swap
	cp -f config.yaml ~/.config/llama-swap/
