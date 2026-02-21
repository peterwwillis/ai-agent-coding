MISE_INSTALL_DIR = $(HOME)/.local/bin

# Add MISE_INSTALL_TOOLS= to Makefile.inc to install a set of tools with mise
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
	@echo "  install			Runs install-mise-*, install-tools"
	@echo "  install-tools			Installs tools listed in MISE_INSTALL_TOOLS ($(MISE_INSTALL_TOOLS))"

install: $(T_INSTALL) install-tools

install-tools:
	set -eux ; \
	for tool in $(MISE_INSTALL_TOOLS) ; do \
		mise use -g $$tool ; \
	done

check-deps-macos:
	for command in tar grep curl xcode-select ; do \
		command -v $$command || exit 1 ; \
	done

install-mise-macos: check-deps-macos
	tmpfile="$$(mktemp)" ; \
	xcode-select --install 2>"$$tmpfile" ; \
	if [ $$? -ne 0 ] ; then \
		if grep 'Command line tools are already installed' "$$tmpfile" ; then \
			true ; \
		else \
			cat "$$tmpfile" ; \
			rm -f "$$tmpfile" ; \
			exit 1 ; \
		fi ; \
		rm -f "$$tmpfile" ; \
	fi
	set -eux ; \
	if ! command -v mise >/dev/null 2>&1 ; then
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
		mkdir -p "$(MISE_INSTALL_DIR)" && \
		mv "$$tmpdir/mise/bin/mise" "$(MISE_INSTALL_DIR)" && \
		chmod 755 "$(MISE_INSTALL_DIR)/mise" && \
		rm -rf "$$tmpdir" && \
		echo "" && \
		echo "Add this to your ~/.bashrc file:" && \
		echo '    eval "$$(mise activate bash)"' ; \
	fi


check-deps-linux:
	for command in sudo add-apt-repository apt ; do \
		command -v $$command || exit 1 ; \
	done

# Install mise system-wide on Linux, partly because it's more secure,
# and partly because system packages are easier to remove/upgrade/etc
install-mise-linux: check-deps-linux
	if ! command -v mise >/dev/null 2>&1 ; then \
		if command -v apt ; then \
			sudo add-apt-repository -y ppa:jdxcode/mise && \
			sudo apt update -y && \
			sudo apt install -y mise && \
			echo "" && \
			echo "Add this to your ~/.bashrc file:" && \
			echo '    eval "$$(mise activate bash)"' ; \
		else \
			exit 1 ; \
		fi ; \
	fi
