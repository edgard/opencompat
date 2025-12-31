// Package api provides OpenAI-compatible API types and utilities.
package api

import "encoding/json"

// ChatCompletionRequest represents an OpenAI chat completion request.
type ChatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []Message       `json:"messages"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	N                   *int            `json:"n,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
	Stop                json.RawMessage `json:"stop,omitempty"` // string or []string
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"` // Newer replacement for max_tokens
	PresencePenalty     *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64        `json:"frequency_penalty,omitempty"`
	LogitBias           map[string]int  `json:"logit_bias,omitempty"`
	User                string          `json:"user,omitempty"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"` // "none", "auto", "required", or object
	ParallelToolCalls   *bool           `json:"parallel_tool_calls,omitempty"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
	Seed                *int            `json:"seed,omitempty"`
	// OpenAI-specific reasoning parameters (passed through)
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

// StreamOptions specifies options for streaming responses.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// Message represents a chat message.
type Message struct {
	Role             string           `json:"role"`
	Content          json.RawMessage  `json:"content"` // string or []ContentPart
	Name             string           `json:"name,omitempty"`
	Refusal          string           `json:"refusal,omitempty"` // Model refusal message
	ToolCalls        []ToolCall       `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	Reasoning        *ReasoningOutput `json:"reasoning,omitempty"`         // For o3 mode
	ReasoningSummary string           `json:"reasoning_summary,omitempty"` // For legacy mode
}

// ContentPart represents a part of a multimodal message.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL represents an image URL in a message.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "low", "high", "auto"
}

// Tool represents a tool/function that can be called.
type Tool struct {
	Type     string   `json:"type"` // "function"
	Function Function `json:"function"`
}

// Function represents a function definition.
type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// ToolCall represents a tool call in a message.
type ToolCall struct {
	Index    *int         `json:"index,omitempty"` // For streaming chunks (pointer so 0 serializes)
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"` // "function"
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call.
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

// ResponseFormat specifies the output format.
type ResponseFormat struct {
	Type string `json:"type"` // "text", "json_object", "json_schema"
}

// ChatCompletionResponse represents an OpenAI chat completion response.
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// Choice represents a completion choice.
type Choice struct {
	Index        int       `json:"index"`
	Message      *Message  `json:"message,omitempty"`
	Delta        *Delta    `json:"delta,omitempty"`
	FinishReason *string   `json:"finish_reason"` // Pointer for proper null serialization
	Logprobs     *Logprobs `json:"logprobs"`      // Always present (null or object)
}

// Logprobs represents log probability information for a choice.
type Logprobs struct {
	Content []LogprobContent `json:"content,omitempty"`
}

// LogprobContent represents log probability for a token.
type LogprobContent struct {
	Token       string  `json:"token"`
	Logprob     float64 `json:"logprob"`
	Bytes       []int   `json:"bytes,omitempty"`
	TopLogprobs []struct {
		Token   string  `json:"token"`
		Logprob float64 `json:"logprob"`
		Bytes   []int   `json:"bytes,omitempty"`
	} `json:"top_logprobs,omitempty"`
}

// Delta represents incremental content in streaming responses.
type Delta struct {
	Role             string           `json:"role,omitempty"`
	Content          string           `json:"content,omitempty"`
	Refusal          string           `json:"refusal,omitempty"` // Model refusal message
	ToolCalls        []ToolCall       `json:"tool_calls,omitempty"`
	Reasoning        *ReasoningOutput `json:"reasoning,omitempty"`         // For o3 mode
	ReasoningSummary string           `json:"reasoning_summary,omitempty"` // For legacy mode
}

// ReasoningOutput represents reasoning content in o3 format.
type ReasoningOutput struct {
	Content []ReasoningContent `json:"content,omitempty"`
}

// ReasoningContent represents a piece of reasoning content.
type ReasoningContent struct {
	Type string `json:"type"` // "text"
	Text string `json:"text,omitempty"`
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens            int                     `json:"prompt_tokens"`
	CompletionTokens        int                     `json:"completion_tokens"`
	TotalTokens             int                     `json:"total_tokens"`
	PromptTokensDetails     *PromptTokenDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokenDetails `json:"completion_tokens_details,omitempty"`
}

// PromptTokenDetails contains detailed breakdown of prompt tokens.
type PromptTokenDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// CompletionTokenDetails contains detailed breakdown of completion tokens.
type CompletionTokenDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// ChatCompletionChunk represents a streaming chunk.
type ChatCompletionChunk struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// ModelsResponse represents the /v1/models response.
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// Model represents a model in the models list.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// GetContentString extracts string content from a message.
// Returns empty string if content is not a simple string.
func (m *Message) GetContentString() string {
	if m.Content == nil {
		return ""
	}

	// Try to unmarshal as string first
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}

	return ""
}

// GetContentParts extracts content parts from a multimodal message.
func (m *Message) GetContentParts() []ContentPart {
	if m.Content == nil {
		return nil
	}

	// Try to unmarshal as array of content parts
	var parts []ContentPart
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		return parts
	}

	// If it's a string, wrap it as a text part
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return []ContentPart{{Type: "text", Text: s}}
	}

	return nil
}

// SetContentString sets the content as a simple string.
func (m *Message) SetContentString(s string) {
	data, _ := json.Marshal(s)
	m.Content = data
}
