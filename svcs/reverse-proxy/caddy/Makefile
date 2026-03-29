
# Relative path to the shared scripts directory
SCRIPTS := ../scripts

.PHONY: caddy-help caddy-setup caddy-up caddy-reload caddy-health caddy-clean check-install gen-certs gen-hash trust-ca

HELP_TARGET = caddy-help
UP_TARGET := caddy-up

caddy-help: help ## Show available make targets
	@echo ""
	@echo "Caddy reverse-proxy – available targets:"
	@echo ""
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "    %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Quickstart:"
	@echo "  1. make caddy-setup        # copy .env.example and generate certs"
	@echo "  2. make gen-hash     # generate a bcrypt hash"
	@echo "  3. \$$EDITOR $(CADDY_CONFIG_DIR)/.env      # paste hash as PROXY_AUTH_HASH, set PROXY_AUTH_USER"
	@echo "  4. make trust-ca     # add CA to OS trust store"
	@echo "  5. make up"
	@echo ""

caddy-up: check-install up ## Start the Caddy stack (validates .env and certs exist first)


caddy-reload: ## Reload Caddy configuration without downtime
	make exec DOCKER_COMPOSE_EXEC_ARGS="caddy reload --config /etc/caddy/Caddyfile"

caddy-health: ## Run health check against the proxy routes
	$(SCRIPTS)/health-check.sh --skip-tls-verify --host "$(PROXY_DOMAIN)"

caddy-install:
	@if [ ! -f $(CADDY_CONFIG_DIR)/Caddyfile ] ; then \
		cp Caddyfile $(CADDY_CONFIG_DIR)/Caddyfile ; \
		echo "Copied Caddyfile to '$(CADDY_CONFIG_DIR)/Caddyfile'" ; \
	else \
		echo "Caddyfile already exists as '$(CADDY_CONFIG_DIR)/Caddyfile', skipping copy" ; \
	fi
	@if [ ! -f $(CADDY_CONFIG_DIR)/.env ]; then \
		cp .env.example $(CADDY_CONFIG_DIR)/.env; \
		echo "Copied .env.example to '$(CADDY_CONFIG_DIR)/.env'" ; \
	else \
		echo ".env already exists as '$(CADDY_CONFIG_DIR)/.env', skipping copy."; \
	fi ; \

caddy-setup: caddy-install ## First-time setup: copy .env.example and generate TLS certs
	echo "Configure $(CADDY_CONFIG_DIR)/.env with your PROXY_DOMAIN entry." ; \
	echo "Then run 'make gen-certs'."

caddy-clean: ## Remove generated certs and .env (DESTRUCTIVE)
	@echo "This will delete $(CADDY_CONFIG_DIR)/certs/ and $(CADDY_CONFIG_DIR)/.env"
	@echo "Press Ctrl-C to cancel, or Enter to continue ..."
	@read -r INPUT
	rm -rf $(CADDY_CONFIG_DIR)/certs/ $(CADDY_CONFIG_DIR)/.env

check-install:
	@[ -f $(CADDY_CONFIG_DIR)/.env ] || { echo "ERROR: .env not found. Run 'make install' first." >&2; exit 1; }
	@HASH="$$(grep -E '^PROXY_AUTH_HASH=' $(CADDY_CONFIG_DIR)/.env | cut -d= -f2-)"; \
	if [ -z "$$HASH" ]; then \
		echo "ERROR: PROXY_AUTH_HASH is not set in .env." >&2; \
		echo "       Run 'make gen-hash' and set it in .env." >&2; exit 1; \
	fi
	@if [ ! -f $(CADDY_CONFIG_DIR)/certs/local.crt ] || [ ! -f $(CADDY_CONFIG_DIR)/certs/local.key ]; then \
		echo "ERROR: $(CADDY_CONFIG_DIR)/certs/local.crt or $(CADDY_CONFIG_DIR)/certs/local.key not found. Run 'make gen-certs' first." >&2; exit 1; \
	fi

gen-certs: ## Generate a CA-signed TLS certificate (uses PROXY_DOMAIN from .env)
	@mkdir -p $(CADDY_CONFIG_DIR)/certs
	$(SCRIPTS)/gen-certs.sh --ca --domain "$(PROXY_DOMAIN)" --out $(CADDY_CONFIG_DIR)/certs/
	@echo ""
	@echo "Next: run 'make gen-hash' to get a bcrypt hash, paste it into .env as"
	@echo "PROXY_AUTH_HASH=..., then run 'make up'."

gen-hash: ## Interactively generate a bcrypt password hash
	@$(SCRIPTS)/gen-bcrypt.sh

## NOTE:
##  - Overwrites file 'local-dev-ca.crt' on Linux,
##    but appends to MacOS keystore.
trust-ca: ## Add the local CA certificate to the OS trust store.
	@if [ ! -f $(CADDY_CONFIG_DIR)/certs/ca.crt ]; then \
		echo "ERROR: $(CADDY_CONFIG_DIR)/certs/ca.crt not found. Run 'make gen-certs' first." >&2; exit 1; \
	fi
	@OS=$$(uname); \
	if [ "$$OS" = "Darwin" ]; then \
		echo "Adding CA to macOS System Keychain (requires sudo)..."; \
		sudo security add-trusted-cert -d -r trustRoot \
			-k /Library/Keychains/System.keychain $(CADDY_CONFIG_DIR)/certs/ca.crt; \
	elif [ -d /usr/local/share/ca-certificates ]; then \
		echo "Adding CA to Debian/Ubuntu trust store (requires sudo)..."; \
		sudo cp $(CADDY_CONFIG_DIR)/certs/ca.crt /usr/local/share/ca-certificates/local-dev-ca.crt; \
		sudo update-ca-certificates; \
	elif [ -d /etc/pki/ca-trust ]; then \
		echo "Adding CA to RHEL/Fedora trust store (requires sudo)..."; \
		sudo cp $(CADDY_CONFIG_DIR)/certs/ca.crt /etc/pki/ca-trust/source/anchors/local-dev-ca.crt; \
		sudo update-ca-trust; \
	else \
		echo "Unknown OS – manually import $(CADDY_CONFIG_DIR)/certs/ca.crt into your browser."; \
	fi


### Load main docker-compose Makefile to override
include ../../../make/docker-compose.Makefile

### Must load the rest after docker-compose.Makefile, as it loads
### the Makefile.inc variables

# Read PROXY_DOMAIN from .env without shell-sourcing (avoids $ expansion in hashes).
_D := $(shell grep -sE '^PROXY_DOMAIN=' $(CADDY_CONFIG_DIR)/.env 2>/dev/null | cut -d= -f2-)
# Defaults to 'localhost'
PROXY_DOMAIN := $(if $(_D),$(_D),localhost)


ifndef CADDY_CONFIG_DIR
$(error Please define CADDY_CONFIG_DIR in Makefile.inc)
endif
