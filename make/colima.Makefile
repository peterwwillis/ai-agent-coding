DEPENDENT_COMMANDS := colima
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
	@echo "  check-deps"
	@echo "  install				Runs 'install-config', 'install-mise'"
	@echo "  install-config"
	@echo "  install-mise"

check-deps:
	@for CMD in $(DEPENDENT_COMMANDS) ; do \
		if ! command -v $$CMD 2>/dev/null 1>/dev/null ; then \
			echo "ERROR: '$$CMD' not found" ; \
			exit 1 ;\
		fi ; \
	done

install: check-deps install-config install-mise

install-config:
	mkdir -p $(CONF_DIR)/$(COLIMA_INSTANCE_NAME)
	if [ ! -e $(CONF_DIR)/$(COLIMA_INSTANCE_NAME)/$(CONF_FILE) ] ; then \
		cp -a $(CONF_FILE) $(CONF_DIR)/$(COLIMA_INSTANCE_NAME)/$(CONF_FILE) ; \
	fi

install-mise:
	mise install colima
