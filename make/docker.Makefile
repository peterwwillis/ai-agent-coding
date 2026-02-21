
# Detect OS, set targets
UNAME_S := $(shell uname -s | tr '[:upper:]' '[:lower:]')
ifeq ($(UNAME_S),linux)
	T_INSTALL = install-linux
endif
ifeq ($(UNAME_S),darwin)
	T_INSTALL = install-macos
endif


install: $(T_INSTALL)

install-linux:
	if ! command -v docker ; then \
		if command -v apt >/dev/null 2>&1; then \
			sudo apt-get update && \
			sudo apt-get install -y ca-certificates curl gnupg && \
			sudo install -m 0755 -d /etc/apt/keyrings && \
			curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
			sudo chmod a+r /etc/apt/keyrings/docker.gpg && \
			echo "deb [arch=$$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $$(. /etc/os-release && echo $$VERSION_CODENAME) stable" | \
				sudo tee /etc/apt/sources.list.d/docker.list > /dev/null && \
			sudo apt-get update && \
			sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin ; \
		else \
			echo "Unsupported Linux distribution" ; \
			exit 1; \
		fi ; \
	fi

install-macos:
	if ! command -v docker ; then \
		if command -v brew >/dev/null 2>&1; then \
			brew install --cask docker; \
		elif command -v port >/dev/null 2>&1; then \
			sudo port install docker docker-compose; \
		else \
			echo "Neither Homebrew nor MacPorts found." ; \
		   	exit 1; \
		fi ; \
	fi
