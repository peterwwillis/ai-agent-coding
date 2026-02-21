COLIMA_CONF_FILE := colima.yaml
LINUX_COLIMA_CONF_DIR := $(HOME)/.config/colima
MACOS_COLIMA_CONF_DIR := $(HOME)/.colima
#
#COLIMA_INSTANCE_NAME := $(notdir $(patsubst %/,%,$(CURDIR)))


####################################################################################

GIT_ROOTDIR := $(shell git rev-parse --show-toplevel)

# Colima depends on qemu and docker
include $(GIT_ROOTDIR)/make/qemu.Makefile
include $(GIT_ROOTDIR)/make/docker.Makefile

# use a Makefile.inc with the following defined:
#     COLIMA_INSTANCE_NAME = <name here>
#
-include Makefile.inc

####################################################################################

# Detect OS, set targets
UNAME_S := $(shell uname -s | tr '[:upper:]' '[:lower:]')
ifeq ($(UNAME_S),linux)
	COLIMA_CONF_DIR = $(LINUX_COLIMA_CONF_DIR)
endif
ifeq ($(UNAME_S),darwin)
	COLIMA_CONF_DIR = $(MACOS_COLIMA_CONF_DIR)
endif
UNAME_M := $(shell uname -m | tr '[:upper:]' '[:lower:]')
ifeq ($(UNAME_M),arm64)
	ARCH = aarch64
endif
ifeq ($(UNAME_M),x86_64)
	ARCH := x86_64
endif

all:
	@echo "Targets:"
	@echo "  install				Runs 'install-mise', 'install-config'"
	@echo "  install-config"
	@echo "  install-mise"

install: install-mise install-config

install-config:
	@if [ -n "$(COLIMA_INSTANCE_NAME)" ] ; then \
		set -x ; \
		mkdir -p $(COLIMA_CONF_DIR)/$(COLIMA_INSTANCE_NAME) && \
		if [ ! -e $(COLIMA_CONF_DIR)/$(COLIMA_INSTANCE_NAME)/$(COLIMA_CONF_FILE) ] ; then \
			cp -a $(COLIMA_CONF_FILE) $(COLIMA_CONF_DIR)/$(COLIMA_INSTANCE_NAME)/$(COLIMA_CONF_FILE) ; \
		fi ; \
		echo "Colima config installed at '$(COLIMA_CONF_DIR)/$(COLIMA_INSTANCE_NAME)/$(COLIMA_CONF_FILE)'." ; \
		echo "Now start your VM with 'colima start $(COLIMA_INSTANCE_NAME)'" ; \
	else \
		echo "WARNING: No COLIMA_INSTANCE_NAME defined; not installing config file!" ; \
	fi

check-deps: install-qemu install-docker
	command -v docker || exit 1
	command -v qemu-system-$(ARCH) || exit 1

install-mise: check-deps
	if ! command -v mise >/dev/null 2>&1 ; then \
		mise use -g lima colima ; \
	fi
