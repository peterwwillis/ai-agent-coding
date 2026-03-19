
%:
	set -eu ; \
	for t in * ; do \
		if [ -d "$$t" ] ; then \
			make -C $$t $@ ; \
		fi ; \
	done
