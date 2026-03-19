
# Add a DOCKER_NETWORK_NAME= to Makefile.inc to create a docker network
-include Makefile.inc

export DOCKER_NETWORK_NAME DOCKER_CONTAINER_NAME DOCKER_BUILD_CONTEXT

all:
	@echo "Targets:"
	@echo "    network				Makes network $(DOCKER_NETWORK_NAME)"
	@echo "    up					Runs 'docker compose up'"
	@echo "    build-devbox			Runs 'docker build -f Dockerfile.devbox'"
	@echo "    shell				Runs 'docker exec -it devbox bash'"

network:
	if [ -n "$(DOCKER_NETWORK_NAME)" ] ; then \
		network_id="$$(docker network ls -q --filter "name=$(DOCKER_NETWORK_NAME)" --filter driver=bridge)" ; \
		if [ -z "$$network_id" ] ; then \
			docker network create --driver bridge --attachable $(DOCKER_NETWORK_NAME) ; \
		fi ; \
	fi

up: network
	export DEVBOX_USER="$$(id -un)" ; \
	export DEVBOX_UID="$$(id -u)" ; \
	export DEVBOX_GID="$$(id -g)" ; \
	docker compose up -d --remove-orphans --build

down:
	docker compose down

shell: up
	docker run --rm -it $(DOCKER_CONTAINER_NAME) bash

build-devbox:
	export DEVBOX_USER="$$(id -un)" ; \
	export DEVBOX_UID="$$(id -u)" ; \
	export DEVBOX_GID="$$(id -g)" ; \
	docker build \
		--progress=plain \
		-f Dockerfile.devbox \
		--build-arg DEVBOX_USER="$$DEVBOX_USER" \
		--build-arg DEVBOX_UID="$$DEVBOX_UID" \
		--build-arg DEVBOX_GID="$$DEVBOX_GID" \
		-t devbox:latest \
		$${DOCKER_BUILD_CONTEXT:-.}

