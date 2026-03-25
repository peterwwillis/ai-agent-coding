# Local dir where you want Caddy certs and .env to go
CADDY_CONFIG_DIR := ~/aia1/svcs/reverse-proxy/caddy

# Runs in the Colima VM
DOCKER_CONTEXT := colima-ai-agent-1
DOCKER_NETWORK_NAME := ai-agent-lab

DOCKER_CONTAINER_NAME := caddy

DOCKER_COMPOSE_ENV_FILE := $(CADDY_CONFIG_DIR)/.env
