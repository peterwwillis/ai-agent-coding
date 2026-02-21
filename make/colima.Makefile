CONF_FILE := colima.yaml
LINUX_CONF_DIR := $(HOME)/.config/colima
MACOS_CONF_DIR := $(HOME)/.colima
COLIMA_INSTANCE_NAME := $(notdir $(patsubst %/,%,$(CURDIR)))

include Makefile.inc

####################################################################################

# Detect OS, set targets
UNAME_S := $(shell uname -s | tr '[:upper:]' '[:lower:]')
ifeq ($(UNAME_S),linux)
	CONF_DIR = $(LINUX_CONF_DIR)
endif
ifeq ($(UNAME_S),darwin)
	CONF_DIR = $(MACOS_CONF_DIR)
endif

all:
	@echo "Targets:"
	@echo "  install				Runs 'install-mise', 'install-config'"
	@echo "  install-config"
	@echo "  install-mise"

install: install-mise install-config

install-config:
	mkdir -p $(CONF_DIR)/$(COLIMA_INSTANCE_NAME)
	if [ ! -e $(CONF_DIR)/$(COLIMA_INSTANCE_NAME)/$(CONF_FILE) ] ; then \
		cp -a $(CONF_FILE) $(CONF_DIR)/$(COLIMA_INSTANCE_NAME)/$(CONF_FILE) ; \
	fi
	@echo "Colima config installed at '$(CONF_DIR)/$(COLIMA_INSTANCE_NAME)/$(CONF_FILE)'."
	@echo "Now start your VM with 'colima start $(COLIMA_INSTANCE_NAME)'"

install-mise:
	mise install colima
