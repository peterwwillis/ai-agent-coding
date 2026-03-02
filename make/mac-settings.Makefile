
all: sshd-config enable-sshd


sshd-config:
	if [ -r sshd_config_user.conf ] ; then \
		cp -v sshd_config_user.conf /etc/ssh/sshd_config.d/100-user.conf ; \
	fi

enable-sshd:
	sudo systemsetup -setremotelogin on
