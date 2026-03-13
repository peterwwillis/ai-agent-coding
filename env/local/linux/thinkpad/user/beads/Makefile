GIT_ROOTDIR := $(shell git rev-parse --show-toplevel)

install: install-dolt
	command -v bd || \
		$(GIT_ROOTDIR)/bin/beads-install.sh

install-dolt:
	command -v dolt || make -C ../dolt/ install
