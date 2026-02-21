
install:
	@if [ "$$(uname -s)" != "Darwin" ]; then \
		echo "MacPorts installer is for macOS only."; exit 1; \
	fi; \
	VERSION=$$(curl -fsSL https://api.github.com/repos/macports/macports-base/releases/latest | grep tag_name | cut -d '"' -f4); \
	PKG="MacPorts-$$VERSION.pkg"; \
	curl -fLO https://github.com/macports/macports-base/releases/download/$$VERSION/$$PKG && \
	sudo installer -pkg $$PKG -target /
