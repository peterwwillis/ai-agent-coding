// Package client provides an OpenAI API client with streaming support.
package client

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// Client wraps the OpenAI client.
type Client struct {
	inner   *openai.Client
	timeout time.Duration
}

// New creates a new Client with the given API key, base URL, and timeout.
func New(apiKey, baseURL string, timeoutSecs int) *Client {
	cfg := openai.DefaultConfig(apiKey)
	if baseURL != "" && baseURL != "default" {
		cfg.BaseURL = baseURL
	}
	return &Client{
		inner:   openai.NewClientWithConfig(cfg),
		timeout: time.Duration(timeoutSecs) * time.Second,
	}
}

// Message is a single chat message for the API.
type Message struct {
	Role    string
	Content string
}

// CompletionRequest holds parameters for a chat completion.
type CompletionRequest struct {
	Model       string
	Messages    []Message
	Temperature float32
	TopP        float32
	Stream      bool
}

// Complete sends a non-streaming completion request and returns the full response.
func (c *Client) Complete(ctx context.Context, req CompletionRequest) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	msgs := toOpenAIMessages(req.Messages)
	resp, err := c.inner.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       req.Model,
		Messages:    msgs,
		Temperature: req.Temperature,
		TopP:        req.TopP,
	})
	if err != nil {
		return "", fmt.Errorf("API error: %w", err)
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no completion returned")
	}
	return resp.Choices[0].Message.Content, nil
}

// StreamComplete sends a streaming completion request, writing tokens to w as they arrive.
// Returns the full accumulated response.
func (c *Client) StreamComplete(ctx context.Context, req CompletionRequest, w io.Writer) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	msgs := toOpenAIMessages(req.Messages)
	stream, err := c.inner.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:       req.Model,
		Messages:    msgs,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      true,
	})
	if err != nil {
		return "", fmt.Errorf("API stream error: %w", err)
	}
	defer stream.Close()

	var sb strings.Builder
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return sb.String(), fmt.Errorf("stream error: %w", err)
		}
		if len(resp.Choices) > 0 {
			token := resp.Choices[0].Delta.Content
			sb.WriteString(token)
			fmt.Fprint(w, token)
		}
	}
	return sb.String(), nil
}

func toOpenAIMessages(msgs []Message) []openai.ChatCompletionMessage {
	out := make([]openai.ChatCompletionMessage, len(msgs))
	for i, m := range msgs {
		out[i] = openai.ChatCompletionMessage{
			Role:    m.Role,
			Content: m.Content,
		}
	}
	return out
}
