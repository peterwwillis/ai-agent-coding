SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets for both proxy stacks
	@echo "Local reverse-proxy – top-level Makefile"
	@echo ""
	@echo "Usage: make <stack>-<target>"
	@echo "  stack  = traefik | caddy"
	@echo "  target = setup | gen-certs | gen-hash | trust-ca | up | down |"
	@echo "           restart | logs | health | clean"
	@echo ""
	@echo "Examples:"
	@echo "  make traefik-setup    # first-time Traefik setup"
	@echo "  make caddy-setup      # first-time Caddy setup"
	@echo "  make traefik-up       # start Traefik"
	@echo "  make caddy-up         # start Caddy"
	@echo "  make traefik-logs     # follow Traefik logs"
	@echo "  make caddy-reload     # reload Caddy config without downtime"
	@echo ""
	@echo "Or cd into a subdirectory for shorter commands:"
	@echo "  cd traefik && make help"
	@echo "  cd caddy   && make help"

# Delegate traefik-<target> to traefik/Makefile
.PHONY: traefik-%
traefik-%:
	$(MAKE) -C traefik $*

# Delegate caddy-<target> to caddy/Makefile
.PHONY: caddy-%
caddy-%:
	$(MAKE) -C caddy $*
