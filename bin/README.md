# bin/

Directory of random useful scripts.


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

   If you want per-model mode presets from `models/model-settings.yml`, add:

   ```bash
   $ ./llama-swap-config-gen.py --verbose --prune-missing --use-model-settings
   ```

4. If it's not already running, run `llama-swap`:

    ```bash
    $ ./llama-swap.sh
    ```

## Estimating llama.cpp parameters

Use `llama-cpp-params.py` to estimate `llama.cpp` flags based on your model and
hardware. The script inspects GGUF metadata (layers, heads, embedding size),
accounts for context/batch sizes and KV cache quantization, and estimates how
many GPU layers will fit in your VRAM (including optional MoE CPU offsets).

```bash
$ ./llama-cpp-params.py \
    --model-path /models/mixtral.gguf \
    --vram 24GB \
    --ctx-size 8192 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --moe-cpu-offset 4
```
