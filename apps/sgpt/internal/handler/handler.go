// Package handler implements the core request/response logic for sgpt.
package handler

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/peterwwillis/ai-agent-coding/apps/sgpt/internal/cache"
	"github.com/peterwwillis/ai-agent-coding/apps/sgpt/internal/chat"
	"github.com/peterwwillis/ai-agent-coding/apps/sgpt/internal/client"
	"github.com/peterwwillis/ai-agent-coding/apps/sgpt/internal/role"
)

// Options configures a completion request.
type Options struct {
	Model       string
	Temperature float32
	TopP        float32
	Caching     bool
	Streaming   bool

	// Mode
	Shell         bool
	Code          bool
	DescribeShell bool
	Interaction   bool // prompt before executing shell command

	// Chat
	ChatSession string // session name or "" for one-shot
	MaxChatLen  int

	// Cache
	CacheDir    string
	CacheMaxLen int
}

// Handler orchestrates a single completion.
type Handler struct {
	client *client.Client
	role   *role.SystemRole
	cache  *cache.Cache
}

// New creates a Handler.
func New(c *client.Client, r *role.SystemRole, cacheDir string, cacheMaxLen int) *Handler {
	return &Handler{
		client: c,
		role:   r,
		cache:  cache.New(cacheDir, cacheMaxLen),
	}
}

// Handle runs a single one-shot or chat completion and returns the response.
func (h *Handler) Handle(ctx context.Context, prompt string, opts Options) (string, error) {
	var messages []client.Message

	if opts.ChatSession != "" {
		// Chat mode: load existing session.
		session, err := chat.Load(opts.ChatSession, opts.MaxChatLen)
		if err != nil {
			return "", err
		}
		// If session is empty, add system message.
		if len(session.Messages) == 0 {
			if err := session.Add("system", h.role.Role); err != nil {
				return "", err
			}
		}
		if err := session.Add("user", prompt); err != nil {
			return "", err
		}
		for _, m := range session.Messages {
			messages = append(messages, client.Message{Role: m.Role, Content: m.Content})
		}

		completion, err := h.complete(ctx, messages, opts)
		if err != nil {
			return "", err
		}
		fmt.Println()

		if err := session.Add("assistant", completion); err != nil {
			return "", err
		}
		return completion, nil
	}

	// Default (one-shot) mode.
	messages = []client.Message{
		{Role: "system", Content: h.role.Role},
		{Role: "user", Content: prompt},
	}

	cacheKey := cache.Key(prompt, opts.Model, h.role.Name, float64(opts.Temperature), float64(opts.TopP))
	if opts.Caching {
		if cached, ok := h.cache.Get(cacheKey); ok {
			fmt.Print(cached)
			fmt.Println()
			return cached, nil
		}
	}

	completion, err := h.complete(ctx, messages, opts)
	if err != nil {
		return "", err
	}
	fmt.Println()

	if opts.Caching {
		_ = h.cache.Set(cacheKey, completion)
	}
	return completion, nil
}

func (h *Handler) complete(ctx context.Context, messages []client.Message, opts Options) (string, error) {
	req := client.CompletionRequest{
		Model:       opts.Model,
		Messages:    messages,
		Temperature: opts.Temperature,
		TopP:        opts.TopP,
	}
	if opts.Streaming {
		return h.client.StreamComplete(ctx, req, os.Stdout)
	}
	completion, err := h.client.Complete(ctx, req)
	if err != nil {
		return "", err
	}
	fmt.Print(completion)
	return completion, nil
}

// RunREPL starts an interactive REPL session.
func RunREPL(ctx context.Context, h *Handler, sessionName string, opts Options, initPrompt string) error {
	session, err := chat.Load(sessionName, opts.MaxChatLen)
	if err != nil {
		return err
	}
	if len(session.Messages) == 0 {
		if err := session.Add("system", h.role.Role); err != nil {
			return err
		}
	}

	fmt.Fprintln(os.Stderr, "Entering REPL mode, press Ctrl+C to exit.")

	if initPrompt != "" {
		if err := processReplPrompt(ctx, h, session, initPrompt, opts); err != nil {
			return err
		}
	}

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print(">>> ")
		if !scanner.Scan() {
			break
		}
		prompt := strings.TrimSpace(scanner.Text())
		if prompt == "" {
			continue
		}
		if opts.Shell && (prompt == "e" || prompt == "E") {
			// Execute the last assistant message as a shell command.
			if len(session.Messages) > 0 {
				last := session.Messages[len(session.Messages)-1]
				if last.Role == "assistant" {
					if err := runShellCommand(last.Content); err != nil {
						fmt.Fprintln(os.Stderr, "Error:", err)
					}
				}
			}
			continue
		}
		if err := processReplPrompt(ctx, h, session, prompt, opts); err != nil {
			fmt.Fprintln(os.Stderr, "Error:", err)
		}
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		return err
	}
	return nil
}

func processReplPrompt(ctx context.Context, h *Handler, session *chat.Session, prompt string, opts Options) error {
	if err := session.Add("user", prompt); err != nil {
		return err
	}

	var messages []client.Message
	for _, m := range session.Messages {
		messages = append(messages, client.Message{Role: m.Role, Content: m.Content})
	}

	req := client.CompletionRequest{
		Model:       opts.Model,
		Messages:    messages,
		Temperature: opts.Temperature,
		TopP:        opts.TopP,
	}

	var completion string
	var err error
	if opts.Streaming {
		completion, err = h.client.StreamComplete(ctx, req, os.Stdout)
	} else {
		completion, err = h.client.Complete(ctx, req)
		if err == nil {
			fmt.Print(completion)
		}
	}
	if err != nil {
		return err
	}
	fmt.Println()

	return session.Add("assistant", completion)
}

// PromptShellAction prompts the user to execute, describe, modify, or abort a shell command.
// Returns the (possibly modified) command and whether the loop should continue.
func PromptShellAction(ctx context.Context, h *Handler, command string, opts Options, defaultExec bool) error {
	reader := bufio.NewReader(os.Stdin)
	for {
		defaultOpt := "a"
		if defaultExec {
			defaultOpt = "e"
		}
		fmt.Printf("[E]xecute, [D]escribe, [A]bort [default: %s]: ", strings.ToUpper(defaultOpt))

		input, err := reader.ReadString('\n')
		if err != nil {
			return nil
		}
		choice := strings.ToLower(strings.TrimSpace(input))
		if choice == "" {
			choice = defaultOpt
		}

		switch choice {
		case "e", "y":
			return runShellCommand(command)
		case "d":
			// Describe the command.
			descRole := role.GetDescribeShell()
			descHandler := New(h.client, descRole, opts.CacheDir, opts.CacheMaxLen)
			_, err := descHandler.Handle(ctx, command, Options{
				Model:       opts.Model,
				Temperature: opts.Temperature,
				TopP:        opts.TopP,
				Caching:     opts.Caching,
				Streaming:   opts.Streaming,
				CacheDir:    opts.CacheDir,
				CacheMaxLen: opts.CacheMaxLen,
			})
			if err != nil {
				fmt.Fprintln(os.Stderr, "Error:", err)
			}
			continue
		case "a":
			return nil
		default:
			fmt.Fprintln(os.Stderr, "Invalid option. Choose [E]xecute, [D]escribe, or [A]bort.")
		}
	}
}

// runShellCommand executes the given command string in the user's shell.
func runShellCommand(command string) error {
	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/sh"
	}
	cmd := exec.Command(shell, "-c", command)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
