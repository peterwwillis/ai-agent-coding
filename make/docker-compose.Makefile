
# Add a DOCKER_NETWORK_NAME= to Makefile.inc to create a docker network
-include Makefile.inc

export DOCKER_NETWORK_NAME DOCKER_CONTAINER_NAME DOCKER_BUILD_CONTEXT

all:
	@echo "Targets:"
	@echo "    network				Makes network $(DOCKER_NETWORK_NAME)"
	@echo "    up					Runs 'docker compose up'"
	@echo "    build				Runs 'docker build -f $(DOCKERFILE)'"
	@echo "    shell				Runs 'docker exec -it $(DOCKER_CONTAINER_NAME) bash'"

network:
	if [ -n "$(DOCKER_NETWORK_NAME)" ] ; then \
		network_id="$$(docker network ls -q --filter "name=$(DOCKER_NETWORK_NAME)" --filter driver=bridge)" ; \
		if [ -z "$$network_id" ] ; then \
			docker network create --driver bridge --attachable $(DOCKER_NETWORK_NAME) ; \
		fi ; \
	fi

up: network
	export USER="$$(id -un)" ; \
	export UID="$$(id -u)" ; \
	export GID="$$(id -g)" ; \
	docker compose up -d --remove-orphans --build

down:
	docker compose down

shell: up
	docker run --rm -it $(DOCKER_CONTAINER_NAME) bash

build:
	export USER="$$(id -un)" ; \
	export UID="$$(id -u)" ; \
	export GID="$$(id -g)" ; \
	docker build \
		--progress=plain \
		-f $(DOCKERFILE) \
		--build-arg USER="$$USER" \
		--build-arg UID="$$UID" \
		--build-arg GID="$$GID" \
		-t $(DOCKER_CONTAINER_NAME):$(DOCKER_CONTAINER_TAG) \
		$${DOCKER_BUILD_CONTEXT:-.}

