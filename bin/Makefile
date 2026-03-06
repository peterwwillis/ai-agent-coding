

update-cache:
	./ollama-to-llamacpp-modelcache.sh -o /usr/share/ollama/.ollama/models
	./llama-swap-config-gen.py --verbose --prune-missing

test:
	poetry run pytest tests/ -v

ruff-format:
	ruff format --line-length 320
