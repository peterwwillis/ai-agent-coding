package cache_test

import (
	"os"
	"testing"

	"github.com/peterwwillis/ai-agent-coding/apps/sgpt/internal/cache"
)

func TestCacheSetGet(t *testing.T) {
	dir := t.TempDir()
	c := cache.New(dir, 10)

	key := cache.Key("hello world", "gpt-4o", "ShellGPT", 0.0, 1.0)
	val := "The answer is 42."

	// Not cached yet.
	if _, ok := c.Get(key); ok {
		t.Error("expected cache miss before Set")
	}

	if err := c.Set(key, val); err != nil {
		t.Fatalf("Set() error: %v", err)
	}

	got, ok := c.Get(key)
	if !ok {
		t.Fatal("expected cache hit after Set")
	}
	if got != val {
		t.Errorf("Get() = %q, want %q", got, val)
	}
}

func TestCacheEviction(t *testing.T) {
	dir := t.TempDir()
	maxLen := 3
	c := cache.New(dir, maxLen)

	// Insert more entries than the max.
	for i := range 5 {
		k := cache.Key(string(rune('a'+i)), "gpt-4o", "ShellGPT", 0, 1)
		if err := c.Set(k, "value"); err != nil {
			t.Fatalf("Set() error: %v", err)
		}
	}

	// Count remaining files.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) > maxLen {
		t.Errorf("expected at most %d cached entries, got %d", maxLen, len(entries))
	}
}

func TestCacheKeyDifferentForDifferentParams(t *testing.T) {
	k1 := cache.Key("prompt", "gpt-4o", "role", 0.0, 1.0)
	k2 := cache.Key("prompt", "gpt-4o", "role", 0.5, 1.0)
	if k1 == k2 {
		t.Error("cache keys should differ when temperature differs")
	}
}
