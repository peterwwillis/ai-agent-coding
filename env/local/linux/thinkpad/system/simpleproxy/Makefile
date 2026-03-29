install:
	set -eu ; \
	if ! command -v simpleproxy ; then \
		sudo apt update ; \
		sudo apt install -y simpleproxy ; \
		sudo cp simpleproxy.service /etc/systemd/system/simpleproxy.service ; \
		sudo systemctl daemon-reload ; \
		sudo systemctl enable --now simpleproxy.service ; \
	fi

