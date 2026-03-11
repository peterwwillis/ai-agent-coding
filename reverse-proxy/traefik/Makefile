SHELL := /bin/bash
.DEFAULT_GOAL := help

# Relative path to the shared scripts directory
SCRIPTS := ../scripts

# Read PROXY_DOMAIN from .env without shell-sourcing (avoids $ expansion in hashes).
_D := $(shell grep -sE '^PROXY_DOMAIN=' .env 2>/dev/null | cut -d= -f2-)
PROXY_DOMAIN := $(if $(_D),$(_D),mac.local)

.PHONY: help
help: ## Show available make targets
	@echo "Traefik reverse-proxy – available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Quickstart:"
	@echo "  1. make setup        # copy .env.example, generate certs + users file"
	@echo "  2. make gen-hash     # generate a bcrypt hash"
	@echo "  3. \$$EDITOR .env      # paste hash as PROXY_AUTH_HASH, set PROXY_AUTH_USER"
	@echo "  4. make gen-users-file"
	@echo "  5. make trust-ca     # add CA to OS trust store"
	@echo "  6. make up"

.PHONY: setup
setup: ## First-time setup: copy .env.example, generate certs and users file
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example – edit it to set your credentials."; \
	else \
		echo ".env already exists, skipping copy."; \
	fi
	@mkdir -p certs secrets
	$(MAKE) gen-certs
	@echo ""
	@echo "Next: run 'make gen-hash' to get a bcrypt hash, paste it into .env as"
	@echo "PROXY_AUTH_HASH=..., then run 'make gen-users-file' and 'make up'."

.PHONY: gen-certs
gen-certs: ## Generate a CA-signed TLS certificate (uses PROXY_DOMAIN from .env)
	@mkdir -p certs
	$(SCRIPTS)/gen-certs.sh --ca --domain "$(PROXY_DOMAIN)" --out certs/

.PHONY: gen-hash
gen-hash: ## Interactively generate a bcrypt password hash
	@$(SCRIPTS)/gen-bcrypt.sh

.PHONY: gen-users-file
gen-users-file: ## Generate Traefik basic-auth users file from .env credentials
	@[ -f .env ] || { echo "ERROR: .env not found. Run 'make setup' first." >&2; exit 1; }
	@PROXY_AUTH_USER="$$(grep -E '^PROXY_AUTH_USER=' .env | cut -d= -f2-)"; \
	PROXY_AUTH_HASH="$$(grep -E '^PROXY_AUTH_HASH=' .env | cut -d= -f2-)"; \
	if [ -z "$$PROXY_AUTH_USER" ]; then \
		echo "ERROR: PROXY_AUTH_USER is not set in .env" >&2; exit 1; \
	fi; \
	if [ -z "$$PROXY_AUTH_HASH" ]; then \
		echo "ERROR: PROXY_AUTH_HASH is not set in .env" >&2; \
		echo "       Run 'make gen-hash' to generate one." >&2; exit 1; \
	fi; \
	mkdir -p secrets; \
	printf '%s:%s\n' "$$PROXY_AUTH_USER" "$$PROXY_AUTH_HASH" > secrets/users; \
	chmod 600 secrets/users; \
	echo "Generated secrets/users for user '$$PROXY_AUTH_USER'."
	@echo "NOTE: secrets/users is git-ignored and must not be committed."

.PHONY: trust-ca
trust-ca: ## Add the local CA certificate to the OS trust store
	@if [ ! -f certs/ca.crt ]; then \
		echo "ERROR: certs/ca.crt not found. Run 'make gen-certs' first." >&2; exit 1; \
	fi
	@OS=$$(uname); \
	if [ "$$OS" = "Darwin" ]; then \
		echo "Adding CA to macOS System Keychain (requires sudo)..."; \
		sudo security add-trusted-cert -d -r trustRoot \
			-k /Library/Keychains/System.keychain certs/ca.crt; \
	elif [ -d /usr/local/share/ca-certificates ]; then \
		echo "Adding CA to Debian/Ubuntu trust store (requires sudo)..."; \
		sudo cp certs/ca.crt /usr/local/share/ca-certificates/local-dev-ca.crt; \
		sudo update-ca-certificates; \
	elif [ -d /etc/pki/ca-trust ]; then \
		echo "Adding CA to RHEL/Fedora trust store (requires sudo)..."; \
		sudo cp certs/ca.crt /etc/pki/ca-trust/source/anchors/local-dev-ca.crt; \
		sudo update-ca-trust; \
	else \
		echo "Unknown OS – manually import certs/ca.crt into your browser."; \
	fi

.PHONY: up
up: ## Start the Traefik stack (validates secrets/users exists first)
	@if [ ! -f secrets/users ]; then \
		echo "ERROR: secrets/users not found. Run 'make gen-users-file' first." >&2; exit 1; \
	fi
	@if [ ! -f certs/local.crt ] || [ ! -f certs/local.key ]; then \
		echo "ERROR: certs/local.crt or certs/local.key not found. Run 'make gen-certs' first." >&2; exit 1; \
	fi
	docker compose up -d

.PHONY: down
down: ## Stop the Traefik stack
	docker compose down

.PHONY: restart
restart: ## Restart the Traefik stack
	docker compose restart

.PHONY: logs
logs: ## Follow Traefik logs
	docker compose logs -f

.PHONY: health
health: ## Run health check against the proxy routes
	$(SCRIPTS)/health-check.sh --skip-tls-verify --host "$(PROXY_DOMAIN)"

.PHONY: clean
clean: ## Remove generated certs, secrets, and .env (DESTRUCTIVE)
	@echo "This will delete certs/, secrets/, and .env. Press Ctrl-C to cancel..."
	@sleep 3
	rm -rf certs/ secrets/ .env
