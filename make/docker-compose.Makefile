
.PHONY: up down restart run shell ps build network logs

ifeq ($(OS),Windows_NT)
	TARGETOS := windows
else ifeq ($(shell uname -s),Linux)
	TARGETOS := linux
else
	TARGETOS := macos
endif


HELP_TARGET := help
UP_TARGET := up
DOWN_TARGET := down
RESTART_TARGET := restart
RUN_TARGET := run
SHELL_TARGET := shell
PS_TARGET := ps
BUILD_TARGET := build
NETWORK_TARGET := network


-include Makefile.inc
export DOCKER_NETWORK_NAME DOCKER_CONTAINER_NAME DOCKER_BUILD_CONTEXT DOCKER_COMPOSE_FILE

# Set docker context if specified
ifneq ($(DOCKER_CONTEXT),)
	DOCKER_ARGS := -c $(DOCKER_CONTEXT)
endif

# Add docker compose file arg if specified
ifneq ($(DOCKER_COMPOSE_FILE),)
	DOCKER_COMPOSE_ARGS += -f $(DOCKER_COMPOSE_FILE)
endif

# Add docker compose --env-file
ifneq ($(DOCKER_COMPOSE_ENV_FILE),)
	DOCKER_COMPOSE_ARGS += --env-file $(DOCKER_COMPOSE_ENV_FILE)
endif

all: $(HELP_TARGET)

help:
	@echo "Targets:"
	@echo "    up					Runs 'docker compose up'"
	@echo "    down				Runs 'docker compose down'"
	@echo "    restart				Runs 'docker compose restart'"
	@echo "    run					Runs 'docker run --rm -it $(DOCKER_CONTAINER_NAME)'"
	@echo "    shell				Runs 'docker exec -it $(DOCKER_CONTAINER_NAME) bash'"
	@echo "    ps					Runs 'docker compose ps'"
	@echo "    logs				Runs 'docker compose logs -f'"
	@echo "    network				Makes docker network ($(DOCKER_NETWORK_NAME))"
	@echo "    build				Runs 'docker compose build"
	@echo "    docker-build			Runs 'docker build"

network:
	if [ -n "$(DOCKER_NETWORK_NAME)" ] ; then \
		network_id="$$(docker network ls -q --filter "name=$(DOCKER_NETWORK_NAME)" --filter driver=bridge)" ; \
		if [ -z "$$network_id" ] ; then \
			docker network create --driver bridge --attachable $(DOCKER_NETWORK_NAME) ; \
		fi ; \
	fi

up: $(NETWORK_TARGET)
	export USER="$$(id -un)" UID="$$(id -u)" GID="$$(id -g)" ; \
	docker $(DOCKER_ARGS) compose $(DOCKER_COMPOSE_ARGS) up -d --remove-orphans $(DOCKER_COMPOSE_UP_ARGS)

down:
	export USER="$$(id -un)" UID="$$(id -u)" GID="$$(id -g)" ; \
	docker $(DOCKER_ARGS) compose $(DOCKER_COMPOSE_ARGS) down $(DOCKER_COMPOSE_DOWN_ARGS)

restart:
	export USER="$$(id -un)" UID="$$(id -u)" GID="$$(id -g)" ; \
	docker $(DOCKER_ARGS) compose $(DOCKER_COMPOSE_ARGS) restart $(DOCKER_COMPOSE_RESTART_ARGS)

run: up
	docker $(DOCKER_ARGS) run --rm -it $(DOCKER_CONTAINER_NAME) $(DOCKER_COMPOSE_RUN_ARGS)

shell: up
	docker $(DOCKER_ARGS) exec -it $(DOCKER_CONTAINER_NAME) sh

exec: up
	docker $(DOCKER_ARGS) exec $(DOCKER_CONTAINER_NAME) $(DOCKER_COMPOSE_EXEC_ARGS)

ps:
	docker $(DOCKER_ARGS) compose $(DOCKER_COMPOSE_ARGS) ps

logs:
	docker $(DOCKER_ARGS) compose $(DOCKER_COMPOSE_ARGS) logs -f $(DOCKER_COMPOSE_LOGS_ARGS)


build:
	docker $(DOCKER_ARGS) compose $(DOCKER_COMPOSE_ARGS) build $(DOCKER_COMPOSE_BUILD_ARGS)

docker-build:
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
