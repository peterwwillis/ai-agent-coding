package role_test

import (
	"os"
	"strings"
	"testing"

	"github.com/peterwwillis/ai-agent-coding/sgpt/internal/role"
)

func TestGetForModeDefault(t *testing.T) {
	r := role.GetForMode(false, false, false, "Linux", "bash")
	if r == nil {
		t.Fatal("expected non-nil role")
	}
	if !strings.Contains(r.Role, "ShellGPT") {
		t.Errorf("default role name should contain ShellGPT, got: %q", r.Role)
	}
}

func TestGetForModeShell(t *testing.T) {
	r := role.GetForMode(true, false, false, "Linux", "bash")
	if r == nil {
		t.Fatal("expected non-nil role")
	}
	if !strings.Contains(strings.ToLower(r.Role), "bash") {
		t.Errorf("shell role should mention the shell, got: %q", r.Role)
	}
}

func TestGetForModeCode(t *testing.T) {
	r := role.GetForMode(false, false, true, "Linux", "bash")
	if r == nil {
		t.Fatal("expected non-nil role")
	}
	if !strings.Contains(r.Role, "code") && !strings.Contains(r.Role, "Code") {
		t.Errorf("code role should mention code, got: %q", r.Role)
	}
}

func TestGetForModeDescribeShell(t *testing.T) {
	r := role.GetForMode(false, true, false, "Linux", "bash")
	if r == nil {
		t.Fatal("expected non-nil role")
	}
	if !strings.Contains(r.Role, "shell command") && !strings.Contains(r.Role, "Shell Command") {
		t.Errorf("describe-shell role should mention shell command, got: %q", r.Role)
	}
}

func TestApplyMarkdown(t *testing.T) {
	defaultRole := role.GetDefault("Linux", "bash")
	if !defaultRole.ApplyMarkdown() {
		t.Error("default role should enable APPLY MARKDOWN")
	}
	codeRole := role.GetCode()
	if codeRole.ApplyMarkdown() {
		t.Error("code role should not enable APPLY MARKDOWN")
	}
}

func TestCreateAndGetRole(t *testing.T) {
	dir := t.TempDir()
	role.Init(dir)

	if err := role.Create("testrole", "Always respond with 'test'."); err != nil {
		t.Fatalf("Create() error: %v", err)
	}

	r, err := role.Get("testrole")
	if err != nil {
		t.Fatalf("Get() error: %v", err)
	}
	if !strings.Contains(r.Role, "test") {
		t.Errorf("role content = %q, want to contain 'test'", r.Role)
	}
}

func TestGetNonExistentRole(t *testing.T) {
	dir := t.TempDir()
	role.Init(dir)

	_, err := role.Get("nonexistent")
	if err == nil {
		t.Error("expected error for non-existent role, got nil")
	}
}

func TestListRoles(t *testing.T) {
	dir := t.TempDir()
	role.Init(dir)

	for _, name := range []string{"role1", "role2"} {
		if err := role.Create(name, "a description"); err != nil {
			t.Fatal(err)
		}
	}

	// Verify the role files exist.
	for _, name := range []string{"role1", "role2"} {
		path := dir + "/" + name + ".json"
		if _, err := os.Stat(path); err != nil {
			t.Errorf("expected role file %s to exist: %v", path, err)
		}
	}
}
