// Package config manages the shell_gpt runtime configuration file.
package config

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	configDir  = ".config/shell_gpt"
	configFile = ".sgptrc"
)

// DefaultConfig holds the default configuration values.
var DefaultConfig = map[string]string{
	"CHAT_CACHE_PATH":       filepath.Join(os.TempDir(), "shell_gpt", "chat_cache"),
	"CACHE_PATH":            filepath.Join(os.TempDir(), "shell_gpt", "cache"),
	"CHAT_CACHE_LENGTH":     "100",
	"CACHE_LENGTH":          "100",
	"REQUEST_TIMEOUT":       "60",
	"DEFAULT_MODEL":         "gpt-4o",
	"DEFAULT_COLOR":         "magenta",
	"DEFAULT_EXECUTE_SHELL_CMD": "false",
	"DISABLE_STREAMING":     "false",
	"CODE_THEME":            "dracula",
	"OPENAI_USE_FUNCTIONS":  "true",
	"SHOW_FUNCTIONS_OUTPUT": "false",
	"API_BASE_URL":          "default",
	"PRETTIFY_MARKDOWN":     "true",
	"USE_LITELLM":           "false",
	"SHELL_INTERACTION":     "true",
	"OS_NAME":               "auto",
	"SHELL_NAME":            "auto",
}

// Config holds the runtime configuration.
type Config struct {
	path   string
	values map[string]string
}

// configPath returns the path to the config file.
func configPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, configDir, configFile)
}

// roleStoragePath returns the path to the role storage directory.
func roleStoragePath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, configDir, "roles")
}

// Load loads or creates the config file, returning a Config instance.
func Load() (*Config, error) {
	path := configPath()
	c := &Config{
		path:   path,
		values: make(map[string]string),
	}

	// Set defaults first.
	for k, v := range DefaultConfig {
		c.values[k] = v
	}
	// Override with OS env vars.
	for k := range DefaultConfig {
		if v, ok := os.LookupEnv(k); ok {
			c.values[k] = v
		}
	}

	// Set role storage path based on config dir.
	if _, ok := c.values["ROLE_STORAGE_PATH"]; !ok {
		c.values["ROLE_STORAGE_PATH"] = roleStoragePath()
	}
	if v, ok := os.LookupEnv("ROLE_STORAGE_PATH"); ok {
		c.values["ROLE_STORAGE_PATH"] = v
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		// Config file doesn't exist; prompt for API key if not in env.
		if c.values["OPENAI_API_KEY"] == "" {
			if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
				c.values["OPENAI_API_KEY"] = apiKey
			} else {
				fmt.Print("Please enter your OpenAI API key: ")
				var key string
				fmt.Scanln(&key)
				key = strings.TrimSpace(key)
				if key == "" {
					return nil, fmt.Errorf("API key is required")
				}
				c.values["OPENAI_API_KEY"] = key
			}
		}
		// Ensure config directory exists.
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			return nil, fmt.Errorf("creating config dir: %w", err)
		}
		if err := c.write(); err != nil {
			return nil, err
		}
		return c, nil
	}

	// Read existing config file.
	if err := c.read(); err != nil {
		return nil, err
	}
	return c, nil
}

// Get returns the config value for key, with env var taking priority.
func (c *Config) Get(key string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return c.values[key]
}

// Set updates a config value.
func (c *Config) Set(key, value string) {
	c.values[key] = value
}

func (c *Config) read() error {
	f, err := os.Open(c.path)
	if err != nil {
		return fmt.Errorf("opening config: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		c.values[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
	}
	// Override file values with environment variables.
	for k := range c.values {
		if v := os.Getenv(k); v != "" {
			c.values[k] = v
		}
	}
	return scanner.Err()
}

func (c *Config) write() error {
	f, err := os.Create(c.path)
	if err != nil {
		return fmt.Errorf("writing config: %w", err)
	}
	defer f.Close()

	for k, v := range c.values {
		if _, err := fmt.Fprintf(f, "%s=%s\n", k, v); err != nil {
			return err
		}
	}
	return nil
}
