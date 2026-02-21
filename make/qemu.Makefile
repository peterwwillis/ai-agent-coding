
# Detect OS, set targets
UNAME_S := $(shell uname -s | tr '[:upper:]' '[:lower:]')
ifeq ($(UNAME_S),linux)
	T_INSTALL = install-linux
endif
ifeq ($(UNAME_S),darwin)
	T_INSTALL = install-macos
endif
UNAME_M := $(shell uname -m | tr '[:upper:]' '[:lower:]')
ifeq ($(UNAME_M),arm64)
	ARCH = aarch64
endif
ifeq ($(UNAME_M),x86_64)
	ARCH := x86_64
endif

install: $(T_INSTALL)

install-linux:
	if ! command -v qemu-system-$(ARCH) ; then \
		if command -v apt >/dev/null 2>&1; then \
			sudo apt-get update && \
			sudo apt-get install -y qemu-system qemu-utils qemu-kvm ; \
		else \
			echo "Unsupported Linux distribution" ; \
			exit 1 ; \
		fi ; \
	fi

install-macos:
	if ! command -v qemu-system-$(ARCH) ; then \
		if command -v brew >/dev/null 2>&1; then \
			brew install qemu ; \
		elif command -v port >/dev/null 2>&1; then \
			sudo port install qemu ; \
		else \
			echo "Neither Homebrew nor MacPorts found." ; \
			exit 1 ; \
		fi ; \
	fi
