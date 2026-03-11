# sgpt — Shell GPT (Go Edition)

A command-line productivity tool powered by AI large language models (LLM).
This is a Go rewrite of [TheR1D/shell_gpt](https://github.com/TheR1D/shell_gpt).

## Installation

```bash
cd sgpt
go build -o sgpt .
# Optionally install to your PATH:
go install .
```

## Configuration

On first run, `sgpt` will prompt for your OpenAI API key (if it is not already
set via the `OPENAI_API_KEY` environment variable). The key—along with all other
settings—is stored in `~/.config/shell_gpt/.sgptrc`.

### Configuration file (`~/.config/shell_gpt/.sgptrc`)

```
OPENAI_API_KEY=your_api_key
API_BASE_URL=default
DEFAULT_MODEL=gpt-4o
CHAT_CACHE_PATH=/tmp/shell_gpt/chat_cache
CHAT_CACHE_LENGTH=100
CACHE_PATH=/tmp/shell_gpt/cache
CACHE_LENGTH=100
REQUEST_TIMEOUT=60
DEFAULT_COLOR=magenta
DEFAULT_EXECUTE_SHELL_CMD=false
DISABLE_STREAMING=false
SHELL_INTERACTION=true
OS_NAME=auto
SHELL_NAME=auto
ROLE_STORAGE_PATH=~/.config/shell_gpt/roles
PRETTIFY_MARKDOWN=true
```

All settings can also be overridden by environment variables with the same name.

## Usage

### Basic query

```bash
sgpt "What is the Fibonacci sequence?"
```

### Piping stdin

```bash
git diff | sgpt "Generate a git commit message for my changes"
docker logs -n 20 my_app | sgpt "Check logs, find errors, provide possible solutions"
```

### Shell command generation (`-s` / `--shell`)

```bash
sgpt -s "find all json files in current folder"
# -> find . -type f -name "*.json"
# -> [E]xecute, [D]escribe, [A]bort [default: A]:
```

Use `--no-interaction` to print the command without prompting:

```bash
sgpt -s --no-interaction "find all json files in current folder"
# -> find . -type f -name "*.json"
```

### Describe a shell command (`-d` / `--describe-shell`)

```bash
sgpt -d "find . -type f -name '*.json'"
```

### Code generation (`-c` / `--code`)

```bash
sgpt -c "solve fizz buzz problem using python"
sgpt -c "solve classic fizz buzz problem using Python" > fizz_buzz.py
```

### Chat mode (`--chat`)

```bash
sgpt --chat conv1 "please remember my favourite number: 4"
sgpt --chat conv1 "what would be my favourite number + 4?"
sgpt --list-chats
sgpt --show-chat conv1
```

### REPL mode (`--repl`)

```bash
sgpt --repl temp
# Entering REPL mode, press Ctrl+C to exit.
# >>> What is REPL?
# ...
```

### Custom roles

```bash
sgpt --create-role json_generator
# Enter role description: Provide only valid JSON as response.
sgpt --role json_generator "random user, password, email, address"
sgpt --list-roles
sgpt --show-role json_generator
```

### Request caching

Completions are cached by default (keyed on prompt + model + temperature).
Use `--no-cache` to bypass the cache for a single request.

```bash
sgpt "what are the colours of a rainbow"          # cached after first call
sgpt "what are the colours of a rainbow" --no-cache  # forces a new API call
```

### Editor mode

```bash
sgpt --editor   # opens $EDITOR; content becomes the prompt on save & exit
```

## Full list of flags

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | | LLM to use (default: gpt-4o) |
| `--temperature` | | Randomness 0.0–2.0 (default: 0.0) |
| `--top-p` | | Top-p sampling 0.0–1.0 (default: 1.0) |
| `--md` | | Prettify markdown output |
| `--no-md` | | Disable markdown output |
| `--editor` | | Open `$EDITOR` for the prompt |
| `--cache` | | Cache completions (default: on) |
| `--no-cache` | | Disable caching |
| `--version` | | Show version |
| `--shell` | `-s` | Generate shell commands |
| `--interaction` | | Interactive [E/D/A] prompt for `--shell` |
| `--no-interaction` | | Skip interactive prompt for `--shell` |
| `--describe-shell` | `-d` | Describe a shell command |
| `--code` | `-c` | Generate code only |
| `--chat` | | Chat session name |
| `--repl` | | Start REPL session |
| `--show-chat` | | Print chat session messages |
| `--list-chats` | `-l` | List all chat sessions |
| `--role` | | Use a named system role |
| `--create-role` | | Create a new role |
| `--show-role` | | Print a role definition |
| `--list-roles` | `-r` | List all roles |

## Compatible API backends

`sgpt` works with any OpenAI-compatible API. Set `API_BASE_URL` in the config
(or via environment variable) to point to a local backend such as
[Ollama](https://github.com/ollama/ollama) or
[llama-swap](https://github.com/mostlygeek/llama-swap):

```bash
export API_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama   # required but can be any non-empty value
export DEFAULT_MODEL=llama3
sgpt "Hello!"
```
