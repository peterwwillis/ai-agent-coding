
build:
	for file in *.Modelfile ; do \
		bn="$$(basename "$$file" .Modelfile)" ; \
		ollama create "$$bn" -f "$$file" ; \
	done
