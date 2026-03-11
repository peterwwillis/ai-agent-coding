package chat_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/peterwwillis/ai-agent-coding/sgpt/internal/chat"
)

func TestChatSessionAddAndLoad(t *testing.T) {
	dir := t.TempDir()
	chat.Init(dir)

	s, err := chat.Load("test-session", 100)
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if len(s.Messages) != 0 {
		t.Errorf("new session should be empty, got %d messages", len(s.Messages))
	}

	if err := s.Add("user", "Hello"); err != nil {
		t.Fatalf("Add() error: %v", err)
	}
	if err := s.Add("assistant", "Hi there!"); err != nil {
		t.Fatalf("Add() error: %v", err)
	}

	// Reload and verify persistence.
	s2, err := chat.Load("test-session", 100)
	if err != nil {
		t.Fatalf("Load() (reload) error: %v", err)
	}
	if len(s2.Messages) != 2 {
		t.Errorf("expected 2 messages after reload, got %d", len(s2.Messages))
	}
	if s2.Messages[0].Content != "Hello" {
		t.Errorf("first message content = %q, want %q", s2.Messages[0].Content, "Hello")
	}
}

func TestChatSessionTrimToMaxLen(t *testing.T) {
	dir := t.TempDir()
	chat.Init(dir)

	s, err := chat.Load("trim-session", 3)
	if err != nil {
		t.Fatal(err)
	}

	for i := range 5 {
		r := "user"
		if i%2 == 1 {
			r = "assistant"
		}
		if err := s.Add(r, "message"); err != nil {
			t.Fatal(err)
		}
	}

	if len(s.Messages) > 3 {
		t.Errorf("expected at most 3 messages, got %d", len(s.Messages))
	}
}

func TestChatSessionFilesExist(t *testing.T) {
	dir := t.TempDir()
	chat.Init(dir)

	for _, name := range []string{"alpha", "beta"} {
		s, err := chat.Load(name, 100)
		if err != nil {
			t.Fatalf("Load(%s) error: %v", name, err)
		}
		if err := s.Add("user", "hi"); err != nil {
			t.Fatal(err)
		}
	}

	for _, name := range []string{"alpha", "beta"} {
		path := filepath.Join(dir, name+".json")
		if _, err := os.Stat(path); err != nil {
			t.Errorf("expected session file %s to exist: %v", path, err)
		}
	}
}

