// Package cache provides request-level caching for API completions.
package cache

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// CacheEntry represents a cached completion.
type CacheEntry struct {
	Completion string `json:"completion"`
}

// Cache manages file-based caching of API completions.
type Cache struct {
	dir    string
	maxLen int
}

// New creates a new Cache that stores results in dir, keeping at most maxLen entries.
func New(dir string, maxLen int) *Cache {
	return &Cache{dir: dir, maxLen: maxLen}
}

// Key computes a cache key from request parameters.
func Key(prompt, model, role string, temperature, topP float64) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%s|%s|%f|%f", prompt, model, role, temperature, topP)
	return fmt.Sprintf("%x", h.Sum(nil))
}

// Get retrieves a cached completion for the given key.
// Returns "", false if not cached.
func (c *Cache) Get(key string) (string, bool) {
	data, err := os.ReadFile(c.entryPath(key))
	if err != nil {
		return "", false
	}
	var entry CacheEntry
	if err := json.Unmarshal(data, &entry); err != nil {
		return "", false
	}
	return entry.Completion, true
}

// Set stores a completion in the cache, evicting old entries if necessary.
func (c *Cache) Set(key, completion string) error {
	if err := os.MkdirAll(c.dir, 0700); err != nil {
		return fmt.Errorf("creating cache dir: %w", err)
	}
	data, err := json.Marshal(CacheEntry{Completion: completion})
	if err != nil {
		return err
	}
	if err := os.WriteFile(c.entryPath(key), data, 0600); err != nil {
		return err
	}
	return c.evict()
}

func (c *Cache) entryPath(key string) string {
	return filepath.Join(c.dir, key+".json")
}

// evict removes the oldest entries when the cache exceeds maxLen.
func (c *Cache) evict() error {
	if c.maxLen <= 0 {
		return nil
	}
	entries, err := os.ReadDir(c.dir)
	if err != nil {
		return nil
	}
	if len(entries) <= c.maxLen {
		return nil
	}

	type fileInfo struct {
		path    string
		modTime int64
	}
	files := make([]fileInfo, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		files = append(files, fileInfo{
			path:    filepath.Join(c.dir, e.Name()),
			modTime: info.ModTime().UnixNano(),
		})
	}
	sort.Slice(files, func(i, j int) bool {
		return files[i].modTime < files[j].modTime
	})
	for i := 0; i < len(files)-c.maxLen; i++ {
		os.Remove(files[i].path)
	}
	return nil
}
