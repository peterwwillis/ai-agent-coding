
# Add a DOCKER_NETWORK_NAME= to Makefile.inc to create a docker network
-include Makefile.inc

all:
	@echo "Targets:"
	@echo "    network				Makes network $(DOCKER_NETWORK_NAME)"
	@echo "    compose-up				Runs 'docker compose up'"
	@echo "    build-devbox			Runs 'docker build -f Dockerfile.devbox'"

network:
	if [ -n "$(DOCKER_NETWORK_NAME)" ] ; then \
		network_id="$$(docker network ls -q --filter "name=$(DOCKER_NETWORK_NAME)" --filter driver=bridge)" ; \
		if [ -z "$$network_id" ] ; then \
			docker network create --driver bridge --attachable dind-lab ; \
		fi ; \
	fi

compose-up: network
	docker compose up


docker-build-devbox:
	docker build -f Dockerfile.devbox -t devbox:latest .

