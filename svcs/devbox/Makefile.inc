# Runs in the Colima VM
DOCKER_CONTEXT := colima-ai-agent-1
DOCKER_NETWORK_NAME = ai-agent-lab

DOCKER_CONTAINER_NAME = devbox
DOCKER_CONTAINER_TAG = latest

DOCKER_BUILD_CONTEXT = ../../
DOCKERFILE = Dockerfile.devbox
