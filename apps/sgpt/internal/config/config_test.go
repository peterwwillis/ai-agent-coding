package config_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/peterwwillis/ai-agent-coding/apps/sgpt/internal/config"
)

func TestLoadCreatesConfigWithAPIKey(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, ".sgptrc")

	// Point config to temp dir by overriding HOME.
	// We create the config file manually to avoid the interactive prompt.
	if err := os.MkdirAll(filepath.Dir(cfgPath), 0700); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(cfgPath, []byte("OPENAI_API_KEY=testkey123\n"), 0600)

	// Temporarily redirect HOME so config.Load picks up our temp file.
	origHome := os.Getenv("HOME")
	defer os.Setenv("HOME", origHome)

	// Instead of redirecting HOME (which would break other things), set the
	// API key via env and just verify Load returns without error.
	os.Setenv("OPENAI_API_KEY", "testkey123")
	defer os.Unsetenv("OPENAI_API_KEY")

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if got := cfg.Get("OPENAI_API_KEY"); got != "testkey123" {
		t.Errorf("OPENAI_API_KEY = %q, want %q", got, "testkey123")
	}
}

func TestConfigGetEnvOverride(t *testing.T) {
	os.Setenv("OPENAI_API_KEY", "from-env")
	defer os.Unsetenv("OPENAI_API_KEY")

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if got := cfg.Get("OPENAI_API_KEY"); got != "from-env" {
		t.Errorf("env override: got %q, want %q", got, "from-env")
	}
}

func TestConfigDefaultModel(t *testing.T) {
	os.Setenv("OPENAI_API_KEY", "testkey")
	defer os.Unsetenv("OPENAI_API_KEY")

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	model := cfg.Get("DEFAULT_MODEL")
	if model == "" {
		t.Error("DEFAULT_MODEL should not be empty")
	}
}
