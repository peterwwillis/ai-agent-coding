INSTALL_DIR = $(HOME)/.local/bin

-include Makefile.inc

####################################################################################

# Detect OS, set targets
UNAME_S := $(shell uname -s | tr '[:upper:]' '[:lower:]')
ifeq ($(UNAME_S),linux)
	T_INSTALL = install-mise-linux
endif
ifeq ($(UNAME_S),darwin)
	T_INSTALL = install-mise-macos
endif

all:
	@echo "Targets:"
	@echo "  install"

install: $(T_INSTALL)

check-deps-macos:
	for command in tar grep curl xcode-select ; do \
		command -v $$command || exit 1 ; \
	done

install-mise-macos: check-deps-macos
	xcode-select --install
	set -eux ; \
	ARCH="$$(uname -m)" ; \
	[ "$$ARCH" = "arm64" ] && TARGET="macos-arm64" ; \
	[ "$$ARCH" = "x86_64" ] && TARGET="macos-x64" ; \
	VERSION="$$(curl -fsSL https://api.github.com/repos/jdx/mise/releases/latest \
	  | grep '"tag_name"' | head -n1 cut -d '"' -f4)" ; \
	BASE_URL="https://github.com/jdx/mise/releases/download" ; \
	ARCHIVE="mise-$$VERSION-$$TARGET.tar.gz" ; \
	CHECKSUMS="SHASUMS256.txt" ; \
	URL="$$BASE_URL/$$ARCHIVE" ; \
	CHECKSUMURL="$$BASE_URL/$$CHECKSUMS" ; \
	curl -fLO "$$URL" && \
	curl -fLO "$$CHECKSUMURL" && \
	grep -E "[[:space:]]+$$ARCHIVE$$" "$$CHECKSUMS" | shasum -a 256 -c - && \
	tmpdir="$$(mktemp -d)" && \
	tar -C "$$tmpdir" -xf "$$ARCHIVE" && \
	mkdir -p "$(INSTALL_DIR)" && \
	mv "$$tmpdir/mise/bin/mise" "$(INSTALL_DIR)" && \
	chmod 755 "$(INSTALL_DIR)/mise" && \
	rm -rf "$$tmpdir" && \
	echo "" && \
	echo "Add this to your ~/.bashrc file:" && \
	echo '    eval "$$(mise activate bash)"'


check-deps-linux:
	for command in sudo add-apt-repository apt ; do \
		command -v $$command || exit 1 ; \
	done

install-mise-linux: check-deps-linux
	if ! command -v mise >/dev/null 2>&1 ; then \
		sudo add-apt-repository -y ppa:jdxcode/mise && \
		sudo apt update -y && \
		sudo apt install -y mise && \
		echo "" && \
		echo "Add this to your ~/.bashrc file:" && \
		echo '    eval "$$(mise activate bash)"' ; \
	fi
