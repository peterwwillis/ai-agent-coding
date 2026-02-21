
NETWORK_NAME = "dind-lab"

-include Makefile.inc

all:
	@echo "Targets:"
	@echo "    network				Makes network $(NETWORK_NAME)"
	@echo "    compose-up				Runs 'docker compose up'"
	@echo "    build-devbox			Runs 'docker build -f Dockerfile.devbox'"

network:
	network_id="$$(docker network ls -q --filter "name=$(NETWORK_NAME)" --filter driver=bridge)" ; \
	if [ -z "$$network_id" ] ; then \
		docker network create --driver bridge --attachable dind-lab ; \
	fi

compose-up: network
	docker compose up


docker-build-devbox:
	docker build -f Dockerfile.devbox -t devbox:latest .

