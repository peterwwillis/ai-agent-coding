GIT_ROOTDIR := $(shell git rev-parse --show-toplevel)

install: install-dolt
	if ! command -v bd >/dev/null ; then \
		$(GIT_ROOTDIR)/bin/beads-install.sh ; \
	fi

install-dolt:
	make -C ../dolt/ install
