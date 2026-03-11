
.PHONY: test test-rag update-cache poetry-install ruff-format

update-cache: poetry-install
	./ollama-to-llamacpp-modelcache.sh -o /usr/share/ollama/.ollama/models
	poetry run ./llama-swap-config-gen.py --verbose --prune-missing

test: poetry-install
	poetry run pytest tests/ -v

test-rag: poetry-install
	poetry run pytest tests/test_linux_doc_rag.py -v

poetry-install:
	poetry install

ruff-format:
	ruff format --line-length 320
