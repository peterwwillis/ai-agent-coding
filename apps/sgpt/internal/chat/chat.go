// Package chat manages persistent chat sessions backed by JSON files.
package chat

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// Message represents a single chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Session holds an ordered list of chat messages.
type Session struct {
	Messages []Message `json:"messages"`
	path     string
	maxLen   int
}

// StoragePath is the directory where chat session files are stored.
var StoragePath string

// Init sets the storage path for chat sessions.
func Init(storagePath string) {
	StoragePath = storagePath
}

// sessionPath returns the file path for the given session name.
func sessionPath(name string) string {
	return filepath.Join(StoragePath, name+".json")
}

// Load loads an existing chat session or returns an empty one.
func Load(name string, maxLen int) (*Session, error) {
	if err := os.MkdirAll(StoragePath, 0700); err != nil {
		return nil, fmt.Errorf("creating chat dir: %w", err)
	}
	path := sessionPath(name)
	s := &Session{path: path, maxLen: maxLen}

	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return s, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading chat session: %w", err)
	}
	if err := json.Unmarshal(data, &s.Messages); err != nil {
		return nil, fmt.Errorf("parsing chat session: %w", err)
	}
	return s, nil
}

// Add appends a message and saves the session.
func (s *Session) Add(role, content string) error {
	s.Messages = append(s.Messages, Message{Role: role, Content: content})
	// Trim to max length (keep system message + last N messages).
	if s.maxLen > 0 && len(s.Messages) > s.maxLen {
		start := 0
		if len(s.Messages) > 0 && s.Messages[0].Role == "system" {
			start = 1
		}
		excess := len(s.Messages) - s.maxLen
		if excess > 0 {
			if start == 1 {
				// Keep system message; remove oldest non-system messages.
				s.Messages = append(s.Messages[:1], s.Messages[1+excess:]...)
			} else {
				s.Messages = s.Messages[excess:]
			}
		}
	}
	return s.save()
}

// OpenAIMessages returns the messages in a format suitable for the OpenAI API.
func (s *Session) OpenAIMessages() []map[string]string {
	out := make([]map[string]string, len(s.Messages))
	for i, m := range s.Messages {
		out[i] = map[string]string{"role": m.Role, "content": m.Content}
	}
	return out
}

func (s *Session) save() error {
	data, err := json.MarshalIndent(s.Messages, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, data, 0600)
}

// Show prints all messages in the session.
func Show(name string) error {
	path := sessionPath(name)
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("chat session %q not found", name)
	}
	var messages []Message
	if err := json.Unmarshal(data, &messages); err != nil {
		return fmt.Errorf("parsing chat session: %w", err)
	}
	for _, m := range messages {
		if m.Role == "system" {
			continue
		}
		fmt.Printf("%s: %s\n", m.Role, m.Content)
	}
	return nil
}

// List prints all available chat session names.
func List() error {
	if _, err := os.Stat(StoragePath); os.IsNotExist(err) {
		return nil
	}
	entries, err := os.ReadDir(StoragePath)
	if err != nil {
		return fmt.Errorf("listing chats: %w", err)
	}
	paths := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			paths = append(paths, filepath.Join(StoragePath, e.Name()))
		}
	}
	sort.Slice(paths, func(i, j int) bool {
		fi, _ := os.Stat(paths[i])
		fj, _ := os.Stat(paths[j])
		if fi == nil || fj == nil {
			return paths[i] < paths[j]
		}
		return fi.ModTime().Before(fj.ModTime())
	})
	for _, p := range paths {
		fmt.Println(p)
	}
	return nil
}
