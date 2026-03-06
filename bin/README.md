
## Syncing Ollama models with Llama.cpp and llama-swap

Let's say you want to run either Ollama or llama.cpp (via llama-swap),
but you don't want two different copies of the models. Here's an easy
way to reuse the models across tools:

1. Pull your models with `ollama pull <model url>`

2. Update the llama.cpp cache with symlinks from ollama's model cache.

   On a Linux host, ollama's models get downloaded to an Ollama user's
   home dir, which is /usr/shre/ollama/.
   
   ```bash
   $ ./ollama-to-llamacpp-modelcache.sh -o /usr/share/ollama/.ollama/models
   ```

3. Update llama-swap config file based on llama.cpp cache.

   ```bash
   $ ./llama-swap-config-gen.py --verbose --prune-missing
   ```

4. If it's not already running, run `llama-swap`:

   ```bash
   $ ./llama-swap.sh
   ```

