
.PHONY: up down build shell

ifeq ($(OS),Windows_NT)
	TARGETOS := windows
else ifeq ($(shell uname -s),Linux)
	TARGETOS := linux
else
	TARGETOS := macos
endif


NETWORK_TARGET := network
UP_TARGET := up
DOWN_TARGET := down
SHELL_TARGET := shell
PS_TARGET := ps
BUILD_TARGET := build


# Add a DOCKER_NETWORK_NAME= to Makefile.inc to create a docker network
-include Makefile.inc


export DOCKER_NETWORK_NAME DOCKER_CONTAINER_NAME DOCKER_BUILD_CONTEXT DOCKER_COMPOSE_FILE

ifneq ($(DOCKER_CONTEXT),)
	DOCKER_ARGS := -c $(DOCKER_CONTEXT)
endif
ifneq ($(DOCKER_COMPOSE_FILE),)
	DOCKER_COMPOSE_ARGS := -f $(DOCKER_COMPOSE_FILE)
endif


all:
	@echo "Targets:"
	@echo "    up					Runs 'docker compose up'"
	@echo "    down				Runs 'docker compose down'"
	@echo "    network				Makes network $(DOCKER_NETWORK_NAME)"
	@echo "    build				Runs 'docker build -f $(DOCKERFILE)'"
	@echo "    shell				Runs 'docker exec -it $(DOCKER_CONTAINER_NAME) bash'"
	@echo "    ps					Runs 'docker compose ps'"

$(NETWORK_TARGET):
	if [ -n "$(DOCKER_NETWORK_NAME)" ] ; then \
		network_id="$$(docker network ls -q --filter "name=$(DOCKER_NETWORK_NAME)" --filter driver=bridge)" ; \
		if [ -z "$$network_id" ] ; then \
			docker network create --driver bridge --attachable $(DOCKER_NETWORK_NAME) ; \
		fi ; \
	fi

$(UP_TARGET): network
	export USER="$$(id -un)" ; \
	export UID="$$(id -u)" ; \
	export GID="$$(id -g)" ; \
	docker $(DOCKER_ARGS) compose $(DOCKER_COMPOSE_ARGS) up -d --remove-orphans --build $(DOCKER_COMPOSE_UP_ARGS)

$(DOWN_TARGET):
	docker $(DOCKER_ARGS) compose $(DOCKER_COMPOSE_ARGS) down $(DOCKER_COMPOSE_DOWN_ARGS)

#$(SHELL_TARGET): up
#	docker $(DOCKER_ARGS) run --rm -it $(DOCKER_CONTAINER_NAME) bash

$(SHELL_TARGET): up
	docker $(DOCKER_ARGS) exec -it $(DOCKER_CONTAINER_NAME) bash

$(PS_TARGET):
	docker $(DOCKER_ARGS) compose $(DOCKER_COMPOSE_ARGS) ps

$(BUILD_TARGET):
	export USER="$$(id -un)" ; \
	export UID="$$(id -u)" ; \
	export GID="$$(id -g)" ; \
	docker $(DOCKER_ARGS) build \
		--progress=plain \
		-f $(DOCKERFILE) \
		--build-arg USER="$$USER" \
		--build-arg UID="$$UID" \
		--build-arg GID="$$GID" \
		-t $(DOCKER_CONTAINER_NAME):$(DOCKER_CONTAINER_TAG) \
		$${DOCKER_BUILD_CONTEXT:-.}

