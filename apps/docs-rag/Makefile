
.PHONY: test test-rag update-cache poetry-install

update-cache: poetry-install

test: poetry-install
	poetry run pytest tests/ -v

poetry-install:
	poetry install

clean:
	rm -rf __pycache__/ .pytest_cache/
