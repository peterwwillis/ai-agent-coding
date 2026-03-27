# Runs on Host Docker daemon, not in a VM
DOCKER_CONTEXT := default

DOCKER_CONTAINER_NAME := localai
DOCKER_CONTAINER_TAG := vulkan

DOCKER_COMPOSE_FILE := compose.linux-vulkan.yml
