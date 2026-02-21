# Instane name same as this directory name
COLIMA_INSTANCE_NAME := $(notdir $(patsubst %/,%,$(CURDIR)))
