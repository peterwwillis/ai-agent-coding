test: test-bin test-apps

test-bin:
	make -C bin/ test

test-apps:
	make -C apps/ test
