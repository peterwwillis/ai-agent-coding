// sgpt is a command-line productivity tool powered by AI large language models.
// It is a Go rewrite of https://github.com/TheR1D/shell_gpt.
package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/peterwwillis/ai-agent-coding/sgpt/internal/chat"
	"github.com/peterwwillis/ai-agent-coding/sgpt/internal/client"
	"github.com/peterwwillis/ai-agent-coding/sgpt/internal/config"
	"github.com/peterwwillis/ai-agent-coding/sgpt/internal/handler"
	"github.com/peterwwillis/ai-agent-coding/sgpt/internal/role"
)

const version = "1.0.0"

func main() {
	if err := rootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}

func rootCmd() *cobra.Command {
	var (
		model         string
		temperature   float64
		topP          float64
		md            bool
		noMd          bool
		shell         bool
		interaction   bool
		noInteraction bool
		describeShell bool
		code          bool
		editor        bool
		caching       bool
		noCache       bool
		chatSession   string
		replSession   string
		showChat      string
		listChats     bool
		roleName      string
		createRole    string
		showRole      string
		listRoles     bool
		showVersion   bool
	)

	cmd := &cobra.Command{
		Use:   "sgpt [prompt]",
		Short: "A command-line productivity tool powered by AI LLMs",
		Long: `sgpt is a command-line tool that generates shell commands, code snippets,
and answers questions using AI language models.

It is a Go rewrite of https://github.com/TheR1D/shell_gpt.`,
		Args:                  cobra.ArbitraryArgs,
		DisableFlagsInUseLine: true,
		SilenceUsage:          true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if showVersion {
				fmt.Printf("sgpt version %s\n", version)
				return nil
			}

			// Load configuration.
			cfg, err := config.Load()
			if err != nil {
				return fmt.Errorf("loading config: %w", err)
			}

			// Initialise sub-packages with config paths.
			role.Init(cfg.Get("ROLE_STORAGE_PATH"))
			chat.Init(cfg.Get("CHAT_CACHE_PATH"))

			// Handle action-only flags that don't require a prompt.
			if listChats {
				return chat.List()
			}
			if listRoles {
				return role.List()
			}
			if showChat != "" {
				return chat.Show(showChat)
			}
			if showRole != "" {
				return role.Show(showRole)
			}
			if createRole != "" {
				fmt.Print("Enter role description: ")
				reader := bufio.NewReader(os.Stdin)
				desc, err := reader.ReadString('\n')
				if err != nil {
					return fmt.Errorf("reading role description: %w", err)
				}
				return role.Create(createRole, strings.TrimSpace(desc))
			}

			// Validate mutually exclusive flags.
			modeCount := 0
			if shell {
				modeCount++
			}
			if describeShell {
				modeCount++
			}
			if code {
				modeCount++
			}
			if modeCount > 1 {
				return fmt.Errorf("only one of --shell, --describe-shell, and --code can be used at a time")
			}
			if chatSession != "" && replSession != "" {
				return fmt.Errorf("--chat and --repl cannot be used together")
			}

			// Collect prompt from positional args.
			prompt := strings.Join(args, " ")

			// Handle stdin.
			stdinPassed := false
			stat, _ := os.Stdin.Stat()
			if (stat.Mode() & os.ModeCharDevice) == 0 {
				stdinPassed = true
				var sb strings.Builder
				scanner := bufio.NewScanner(os.Stdin)
				for scanner.Scan() {
					line := scanner.Text()
					if strings.Contains(line, "__sgpt__eof__") {
						break
					}
					sb.WriteString(line + "\n")
				}
				stdinText := sb.String()
				if prompt != "" {
					prompt = stdinText + "\n\n" + prompt
				} else {
					prompt = stdinText
				}
				// Reopen stdin for interactive use.
				tty, ttyErr := os.Open("/dev/tty")
				if ttyErr == nil {
					os.Stdin = tty
				}
			}

			// Handle --editor flag.
			if editor {
				if stdinPassed {
					return fmt.Errorf("--editor cannot be used with stdin input")
				}
				editedPrompt, err := getEditorPrompt()
				if err != nil {
					return err
				}
				prompt = editedPrompt
			}

			if prompt == "" && replSession == "" {
				return fmt.Errorf("prompt is required (provide as argument, stdin, or use --repl)")
			}

			// Resolve effective flags.
			effectiveModel := coalesceStr(model, cfg.Get("DEFAULT_MODEL"), "gpt-4o")
			effectiveTemp := float32(temperature)
			effectiveTopP := float32(topP)

			disableStreaming := strings.EqualFold(cfg.Get("DISABLE_STREAMING"), "true")
			streaming := !disableStreaming

			// Markdown: --md / --no-md take precedence over config.
			useMd := strings.EqualFold(cfg.Get("PRETTIFY_MARKDOWN"), "true")
			if cmd.Flags().Changed("md") {
				useMd = md
			}
			if cmd.Flags().Changed("no-md") || noMd {
				useMd = false
			}
			_ = useMd // markdown rendering reserved for future enhancement

			useInteraction := strings.EqualFold(cfg.Get("SHELL_INTERACTION"), "true")
			if noInteraction {
				useInteraction = false
			}
			if cmd.Flags().Changed("interaction") {
				useInteraction = interaction
			}

			defaultExecShell := strings.EqualFold(cfg.Get("DEFAULT_EXECUTE_SHELL_CMD"), "true")

			cachePath := cfg.Get("CACHE_PATH")
			cacheMaxLen := 100
			if v, err := strconv.Atoi(cfg.Get("CACHE_LENGTH")); err == nil {
				cacheMaxLen = v
			}
			chatCacheMaxLen := 100
			if v, err := strconv.Atoi(cfg.Get("CHAT_CACHE_LENGTH")); err == nil {
				chatCacheMaxLen = v
			}

			useCaching := true
			if cmd.Flags().Changed("no-cache") || noCache {
				useCaching = false
			}
			if cmd.Flags().Changed("cache") {
				useCaching = caching
			}

			// Build the API client.
			apiKey := cfg.Get("OPENAI_API_KEY")
			if apiKey == "" {
				return fmt.Errorf("OPENAI_API_KEY is not set")
			}
			timeoutSecs := 60
			if v, err := strconv.Atoi(cfg.Get("REQUEST_TIMEOUT")); err == nil {
				timeoutSecs = v
			}
			apiClient := client.New(apiKey, cfg.Get("API_BASE_URL"), timeoutSecs)

			// Resolve role.
			var sysRole *role.SystemRole
			if roleName != "" {
				sysRole, err = role.Get(roleName)
				if err != nil {
					return err
				}
			} else {
				sysRole = role.GetForMode(shell, describeShell, code,
					cfg.Get("OS_NAME"), cfg.Get("SHELL_NAME"))
			}

			opts := handler.Options{
				Model:         effectiveModel,
				Temperature:   effectiveTemp,
				TopP:          effectiveTopP,
				Caching:       useCaching,
				Streaming:     streaming,
				Shell:         shell,
				Code:          code,
				DescribeShell: describeShell,
				Interaction:   useInteraction,
				MaxChatLen:    chatCacheMaxLen,
				CacheDir:      cachePath,
				CacheMaxLen:   cacheMaxLen,
			}

			h := handler.New(apiClient, sysRole, cachePath, cacheMaxLen)

			// REPL mode.
			if replSession != "" {
				return handler.RunREPL(cmd.Context(), h, replSession, opts, prompt)
			}

			// Chat or one-shot mode.
			if chatSession != "" {
				opts.ChatSession = chatSession
			}

			completion, err := h.Handle(cmd.Context(), prompt, opts)
			if err != nil {
				return err
			}

			// Shell interaction loop.
			if shell && useInteraction {
				if err := handler.PromptShellAction(cmd.Context(), h, completion, opts, defaultExecShell); err != nil {
					fmt.Fprintln(os.Stderr, "Error:", err)
				}
			}

			return nil
		},
	}

	// General options.
	cmd.Flags().StringVar(&model, "model", "", "Large language model to use")
	cmd.Flags().Float64Var(&temperature, "temperature", 0.0, "Randomness of generated output (0.0–2.0)")
	cmd.Flags().Float64Var(&topP, "top-p", 1.0, "Limits highest probable tokens (0.0–1.0)")
	cmd.Flags().BoolVar(&md, "md", false, "Prettify markdown output")
	cmd.Flags().BoolVar(&noMd, "no-md", false, "Disable markdown output")
	cmd.Flags().BoolVar(&editor, "editor", false, "Open $EDITOR to provide a prompt")
	cmd.Flags().BoolVar(&caching, "cache", false, "Cache completion results")
	cmd.Flags().BoolVar(&noCache, "no-cache", false, "Disable caching of completion results")
	cmd.Flags().BoolVar(&showVersion, "version", false, "Show version")

	// Assistance options.
	cmd.Flags().BoolVarP(&shell, "shell", "s", false, "Generate and execute shell commands")
	cmd.Flags().BoolVar(&interaction, "interaction", false, "Interactive mode for --shell")
	cmd.Flags().BoolVar(&noInteraction, "no-interaction", false, "Disable interactive mode for --shell")
	cmd.Flags().BoolVarP(&describeShell, "describe-shell", "d", false, "Describe a shell command")
	cmd.Flags().BoolVarP(&code, "code", "c", false, "Generate only code")

	// Chat options.
	cmd.Flags().StringVar(&chatSession, "chat", "", `Follow conversation with id (use "temp" for quick session)`)
	cmd.Flags().StringVar(&replSession, "repl", "", "Start a REPL (Read–eval–print loop) session")
	cmd.Flags().StringVar(&showChat, "show-chat", "", "Show all messages from provided chat id")
	cmd.Flags().BoolVarP(&listChats, "list-chats", "l", false, "List all existing chat ids")

	// Role options.
	cmd.Flags().StringVar(&roleName, "role", "", "System role for the model")
	cmd.Flags().StringVar(&createRole, "create-role", "", "Create a new role")
	cmd.Flags().StringVar(&showRole, "show-role", "", "Show a role")
	cmd.Flags().BoolVarP(&listRoles, "list-roles", "r", false, "List all roles")

	return cmd
}

// coalesceStr returns the first non-empty string.
func coalesceStr(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// getEditorPrompt opens $EDITOR for the user to write a prompt.
func getEditorPrompt() (string, error) {
	editorBin := os.Getenv("EDITOR")
	if editorBin == "" {
		editorBin = "vi"
	}
	f, err := os.CreateTemp("", "sgpt-prompt-*.txt")
	if err != nil {
		return "", fmt.Errorf("creating temp file: %w", err)
	}
	tmpPath := f.Name()
	f.Close()
	defer os.Remove(tmpPath)

	cmd := exec.Command(editorBin, tmpPath)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("editor exited with error: %w", err)
	}

	content, err := os.ReadFile(tmpPath)
	if err != nil {
		return "", fmt.Errorf("reading editor output: %w", err)
	}
	prompt := strings.TrimSpace(string(content))
	if prompt == "" {
		return "", fmt.Errorf("no prompt provided in editor")
	}
	return prompt, nil
}
