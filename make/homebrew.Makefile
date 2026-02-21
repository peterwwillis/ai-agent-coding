
install:
	@if [ "$$(uname -s)" != "Darwin" ]; then \
		echo "Homebrew installer is for macOS only."; exit 1; \
	fi
	/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
