TARGETS := beads colima dolt llama-swap mise-agents mise-misc-tools remmina sgpt sshfs

%:
	for t in $(TARGETS) ; do \
		make -C $$t $@ ; \
	done
