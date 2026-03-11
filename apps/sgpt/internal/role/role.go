// Package role manages system roles for the sgpt application.
package role

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

const roleTemplate = "You are %s\n%s"

// ShellRoleTemplate is the system prompt for shell command generation.
const ShellRoleTemplate = `Provide only %s commands for %s without any description.
If there is a lack of details, provide most logical solution.
Ensure the output is a valid shell command.
If multiple steps required try to combine them together using &&.
Provide only plain text without Markdown formatting.
Do not provide markdown formatting such as ` + "```" + `.`

// DescribeShellRole is the system prompt for describing shell commands.
const DescribeShellRole = `Provide a terse, single sentence description of the given shell command.
Describe each argument and option of the command.
Provide short responses in about 80 words.
APPLY MARKDOWN formatting when possible.`

// CodeRole is the system prompt for code generation.
const CodeRole = `Provide only code as output without any description.
Provide only code in plain text format without Markdown formatting.
Do not include symbols such as ` + "```" + ` or ` + "```" + `python.
If there is a lack of details, provide most logical solution.
You are not allowed to ask for more details.
For example if the prompt is "Hello world Python", you should return "print('Hello world')".`

// DefaultRoleTemplate is the system prompt for the default role.
const DefaultRoleTemplate = `You are programming and system administration assistant.
You are managing %s operating system with %s shell.
Provide short responses in about 100 words, unless you are specifically asked for more details.
If you need to store any data, assume it will be stored in the conversation.
APPLY MARKDOWN formatting when possible.`

// SystemRole represents a named system role with a prompt.
type SystemRole struct {
	Name string `json:"name"`
	Role string `json:"role"`
}

// StoragePath is the directory where role JSON files are stored.
var StoragePath string

// Init sets the storage path for roles.
func Init(storagePath string) {
	StoragePath = storagePath
}

// osName returns the operating system name.
func osName(override string) string {
	if override != "" && override != "auto" {
		return override
	}
	switch runtime.GOOS {
	case "linux":
		return "Linux"
	case "darwin":
		return "Darwin/macOS"
	case "windows":
		return "Windows"
	default:
		return runtime.GOOS
	}
}

// shellName returns the current shell name.
func shellName(override string) string {
	if override != "" && override != "auto" {
		return override
	}
	if runtime.GOOS == "windows" {
		if os.Getenv("PSModulePath") != "" {
			return "powershell.exe"
		}
		return "cmd.exe"
	}
	shell := os.Getenv("SHELL")
	if shell == "" {
		return "sh"
	}
	return filepath.Base(shell)
}

// GetDefault returns the default system role.
func GetDefault(osOverride, shellOverride string) *SystemRole {
	os_ := osName(osOverride)
	sh := shellName(shellOverride)
	return &SystemRole{
		Name: "ShellGPT",
		Role: fmt.Sprintf(roleTemplate, "ShellGPT", fmt.Sprintf(DefaultRoleTemplate, os_, sh)),
	}
}

// GetShell returns the shell command generator role.
func GetShell(osOverride, shellOverride string) *SystemRole {
	os_ := osName(osOverride)
	sh := shellName(shellOverride)
	return &SystemRole{
		Name: "Shell Command Generator",
		Role: fmt.Sprintf(roleTemplate, "Shell Command Generator", fmt.Sprintf(ShellRoleTemplate, sh, os_)),
	}
}

// GetDescribeShell returns the shell command descriptor role.
func GetDescribeShell() *SystemRole {
	return &SystemRole{
		Name: "Shell Command Descriptor",
		Role: fmt.Sprintf(roleTemplate, "Shell Command Descriptor", DescribeShellRole),
	}
}

// GetCode returns the code generator role.
func GetCode() *SystemRole {
	return &SystemRole{
		Name: "Code Generator",
		Role: fmt.Sprintf(roleTemplate, "Code Generator", CodeRole),
	}
}

// GetForMode returns the appropriate role based on mode flags.
func GetForMode(shell, describeShell, code bool, osOverride, shellOverride string) *SystemRole {
	switch {
	case shell:
		return GetShell(osOverride, shellOverride)
	case describeShell:
		return GetDescribeShell()
	case code:
		return GetCode()
	default:
		return GetDefault(osOverride, shellOverride)
	}
}

// filePath returns the path for a named role file.
func filePath(name string) string {
	return filepath.Join(StoragePath, name+".json")
}

// Get loads a custom role by name.
func Get(name string) (*SystemRole, error) {
	path := filePath(name)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("role %q not found", name)
	}
	var r SystemRole
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("parsing role %q: %w", name, err)
	}
	return &r, nil
}

// Create creates a new custom role.
func Create(name, description string) error {
	if err := os.MkdirAll(StoragePath, 0700); err != nil {
		return fmt.Errorf("creating role storage: %w", err)
	}
	path := filePath(name)
	if _, err := os.Stat(path); err == nil {
		fmt.Printf("Role %q already exists. Overwrite? [y/N]: ", name)
		var answer string
		fmt.Scanln(&answer)
		if !strings.EqualFold(strings.TrimSpace(answer), "y") {
			return fmt.Errorf("aborted")
		}
	}
	r := SystemRole{
		Name: name,
		Role: fmt.Sprintf(roleTemplate, name, description),
	}
	data, err := json.Marshal(r)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

// Show prints the role content to stdout.
func Show(name string) error {
	r, err := Get(name)
	if err != nil {
		return err
	}
	fmt.Println(r.Role)
	return nil
}

// List prints all available role names.
func List() error {
	if _, err := os.Stat(StoragePath); os.IsNotExist(err) {
		return nil
	}
	entries, err := os.ReadDir(StoragePath)
	if err != nil {
		return fmt.Errorf("listing roles: %w", err)
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

// ApplyMarkdown reports whether the role enables markdown formatting.
func (r *SystemRole) ApplyMarkdown() bool {
	return strings.Contains(r.Role, "APPLY MARKDOWN")
}
