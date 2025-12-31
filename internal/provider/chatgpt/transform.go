package chatgpt

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"strings"
	"time"

	"github.com/edgard/opencompat/internal/api"
	"github.com/edgard/opencompat/internal/sse"
)

// intPtr returns a pointer to an int value.
func intPtr(i int) *int {
	return &i
}

// stringPtr returns a pointer to a string value.
func stringPtr(s string) *string {
	return &s
}

// TransformRequest converts an OpenAI chat completion request to a ChatGPT Responses API request.
func TransformRequest(req *api.ChatCompletionRequest, instructions string, cfg *Config) (*ResponsesRequest, error) {
	// Normalize model name and extract effort suffix if present
	model, modelEffort := NormalizeModelNameWithEffort(req.Model)

	// Transform messages to input items
	input, err := transformMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	// Strip IDs from input items for stateless operation
	input = stripInputIDs(input)

	// Transform tools
	var tools []ToolDef
	if len(req.Tools) > 0 {
		tools = transformTools(req.Tools)
	}

	// Determine reasoning effort (priority: model suffix > request param > config)
	effort := cfg.ReasoningEffort
	if req.ReasoningEffort != "" {
		effort = req.ReasoningEffort
	}
	if modelEffort != "" {
		effort = modelEffort
	}
	effort = NormalizeReasoningEffort(model, effort)

	// Generate prompt cache key
	cacheKey := generateCacheKey(instructions, model)

	// Log warnings for unsupported parameters that are silently ignored
	logUnsupportedParams(req)

	// Build the request
	respReq := &ResponsesRequest{
		Model:             model,
		Instructions:      instructions,
		Input:             input,
		Tools:             tools,
		ToolChoice:        req.ToolChoice,
		ParallelToolCalls: req.ParallelToolCalls,
		Store:             false,
		Stream:            true, // Always stream, we'll buffer for non-streaming
		Reasoning: &ReasoningConfig{
			Effort:  effort,
			Summary: cfg.ReasoningSummary,
		},
		Text: &TextConfig{
			Verbosity: cfg.TextVerbosity,
		},
		Include:        []string{"reasoning.encrypted_content"},
		PromptCacheKey: cacheKey,
	}

	// Pass through supported sampling parameters
	if req.Temperature != nil {
		respReq.Temperature = req.Temperature
	}
	if req.TopP != nil {
		respReq.TopP = req.TopP
	}
	// MaxCompletionTokens is the newer name, MaxTokens is legacy
	if req.MaxCompletionTokens != nil {
		respReq.MaxOutputTokens = req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		respReq.MaxOutputTokens = req.MaxTokens
	}
	// Pass through stop sequences if present and not null/empty
	// req.Stop is json.RawMessage, so check for actual content beyond null/[]
	if len(req.Stop) > 0 {
		stopStr := string(req.Stop)
		if stopStr != "null" && stopStr != "[]" {
			respReq.Stop = req.Stop
		}
	}

	return respReq, nil
}

// logUnsupportedParams logs warnings for request parameters that are not supported
// by the ChatGPT Responses API and will be silently ignored.
func logUnsupportedParams(req *api.ChatCompletionRequest) {
	if req.N != nil && *req.N > 1 {
		slog.Warn("parameter not supported by ChatGPT Responses API, ignored",
			"param", "n",
			"value", *req.N,
			"note", "only n=1 is supported")
	}
	if req.PresencePenalty != nil && *req.PresencePenalty != 0 {
		slog.Warn("parameter not supported by ChatGPT Responses API, ignored",
			"param", "presence_penalty",
			"value", *req.PresencePenalty)
	}
	if req.FrequencyPenalty != nil && *req.FrequencyPenalty != 0 {
		slog.Warn("parameter not supported by ChatGPT Responses API, ignored",
			"param", "frequency_penalty",
			"value", *req.FrequencyPenalty)
	}
	if len(req.LogitBias) > 0 {
		slog.Warn("parameter not supported by ChatGPT Responses API, ignored",
			"param", "logit_bias")
	}
	if req.Seed != nil {
		slog.Warn("parameter not supported by ChatGPT Responses API, ignored",
			"param", "seed",
			"value", *req.Seed)
	}
	if req.ResponseFormat != nil {
		slog.Warn("parameter not supported by ChatGPT Responses API, ignored",
			"param", "response_format",
			"type", req.ResponseFormat.Type,
			"note", "structured output not supported")
	}
}

// stripInputIDs removes IDs from input items for stateless operation.
// The ChatGPT backend requires store=false for stateless operation, which
// means we can't reference previous items by ID.
func stripInputIDs(input []InputItem) []InputItem {
	result := make([]InputItem, 0, len(input))
	for _, item := range input {
		// Skip item_reference types (server-side state lookup not available in stateless mode)
		if item.Type == "item_reference" {
			continue
		}

		// Copy item without ID (but keep CallID for function_call_output correlation)
		newItem := InputItem{
			Type:      item.Type,
			Role:      item.Role,
			Content:   item.Content,
			CallID:    item.CallID,
			Name:      item.Name,
			Arguments: item.Arguments,
			Output:    item.Output,
			Status:    item.Status,
		}
		// Note: We intentionally don't copy item.ID to strip it
		result = append(result, newItem)
	}
	return result
}

func transformMessages(messages []api.Message) ([]InputItem, error) {
	var input []InputItem

	// First pass: extract system messages and convert to user message
	// The ChatGPT Responses API doesn't support system messages directly,
	// so we convert them to a user message at the start of the conversation
	var systemContent string
	var nonSystemMessages []api.Message
	for _, msg := range messages {
		if msg.Role == "system" {
			content := msg.GetContentString()
			if content != "" {
				if systemContent != "" {
					systemContent += "\n"
				}
				systemContent += content
			}
		} else {
			nonSystemMessages = append(nonSystemMessages, msg)
		}
	}

	// Prepend system content as a user message if present
	if systemContent != "" {
		contentJSON, _ := json.Marshal(systemContent)
		input = append(input, InputItem{
			Type:    "message",
			Role:    "user",
			Content: contentJSON,
		})
	}

	// Second pass: process non-system messages
	for _, msg := range nonSystemMessages {
		// Handle tool results - create function_call_output WITHOUT role field
		// The Responses API expects: {"type": "function_call_output", "call_id": "...", "output": "..."}
		// NOT: {"type": "function_call_output", "role": "tool", ...}
		if msg.Role == "tool" {
			input = append(input, InputItem{
				Type:   "function_call_output",
				CallID: msg.ToolCallID,
				Output: msg.GetContentString(),
			})
			continue
		}

		item := InputItem{
			Type: "message",
			Role: msg.Role,
		}

		// Handle assistant messages with tool calls
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			// First add the assistant message if there's content
			// Check for string content first, then array content
			content := msg.GetContentString()
			if content != "" {
				item.Content, _ = json.Marshal(content)
				input = append(input, item)
			} else {
				// Try array content (multimodal)
				parts := msg.GetContentParts()
				if len(parts) > 0 {
					// Convert to content blocks
					var blocks []ContentBlock
					for _, part := range parts {
						switch part.Type {
						case "text":
							blocks = append(blocks, ContentBlock{
								Type: "input_text",
								Text: part.Text,
							})
						case "image_url":
							if part.ImageURL != nil {
								block := ContentBlock{
									Type:     "input_image",
									ImageURL: part.ImageURL.URL,
								}
								if part.ImageURL.Detail != "" {
									block.Detail = part.ImageURL.Detail
								}
								blocks = append(blocks, block)
							}
						}
					}
					if len(blocks) > 0 {
						item.Content, _ = json.Marshal(blocks)
						input = append(input, item)
					}
				}
			}

			// Then add each function call
			for _, tc := range msg.ToolCalls {
				fcItem := InputItem{
					Type:      "function_call",
					ID:        tc.ID,
					CallID:    tc.ID,
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
					Status:    "completed",
				}
				input = append(input, fcItem)
			}
			continue
		}

		// Handle regular messages (string or multimodal content)
		parts := msg.GetContentParts()
		if len(parts) == 0 {
			// Empty message
			item.Content, _ = json.Marshal("")
			input = append(input, item)
			continue
		}

		// Check if it's a simple text message
		if len(parts) == 1 && parts[0].Type == "text" {
			item.Content, _ = json.Marshal(parts[0].Text)
			input = append(input, item)
			continue
		}

		// Multimodal content - transform to content blocks
		var blocks []ContentBlock
		for _, part := range parts {
			switch part.Type {
			case "text":
				blocks = append(blocks, ContentBlock{
					Type: "input_text",
					Text: part.Text,
				})
			case "image_url":
				if part.ImageURL != nil {
					block := ContentBlock{
						Type:     "input_image",
						ImageURL: part.ImageURL.URL,
					}
					if part.ImageURL.Detail != "" {
						block.Detail = part.ImageURL.Detail
					}
					blocks = append(blocks, block)
				}
			default:
				// Log warning for unknown content part types
				slog.Warn("unknown content part type ignored", "type", part.Type)
			}
		}

		if len(blocks) > 0 {
			item.Content, _ = json.Marshal(blocks)
		}
		input = append(input, item)
	}

	return input, nil
}

func transformTools(tools []api.Tool) []ToolDef {
	var result []ToolDef
	for _, t := range tools {
		if t.Type != "function" {
			continue
		}
		result = append(result, ToolDef{
			Type:        "function",
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
			Strict:      t.Function.Strict,
		})
	}
	return result
}

func generateCacheKey(instructions, model string) string {
	h := sha256.New()
	h.Write([]byte(instructions))
	h.Write([]byte(model))
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// StreamState tracks state during SSE streaming.
type StreamState struct {
	ResponseID            string
	Model                 string
	Created               int64
	CurrentContent        string
	Refusal               string // Model refusal message
	ReasoningSummary      string
	ReasoningFull         string
	ToolCalls             map[int]*api.ToolCall // indexed by output_index
	NextToolIndex         int                   // Next available tool call index
	FinishReason          string
	IncompleteReason      string // "max_output_tokens", "content_filter", etc.
	Usage                 *api.Usage
	ReasoningCompat       string // "none", "think-tags", "o3", "legacy"
	ThinkTagOpen          bool
	ThinkTagClosed        bool
	SawOutput             bool
	SentStopChunk         bool
	PendingSummaryNewline bool
	ErrorMessage          string
	// Web search state tracking (like ChatMock's ws_state/ws_index)
	WebSearchState map[string]*WebSearchAccum // call_id -> accumulated params
	WebSearchIndex map[string]int             // call_id -> output_index
}

// WebSearchAccum accumulates web search parameters across streaming events.
type WebSearchAccum struct {
	Query      string   `json:"query,omitempty"`
	Domains    []string `json:"domains,omitempty"`
	MaxResults int      `json:"max_results,omitempty"`
	Recency    string   `json:"recency,omitempty"`
}

// NewStreamState creates a new stream state.
func NewStreamState() *StreamState {
	return &StreamState{
		ToolCalls:       make(map[int]*api.ToolCall),
		WebSearchState:  make(map[string]*WebSearchAccum),
		WebSearchIndex:  make(map[string]int),
		ReasoningCompat: "none", // Default to none
	}
}

// SetReasoningCompat sets the reasoning compatibility mode.
func (s *StreamState) SetReasoningCompat(mode string) {
	s.ReasoningCompat = mode
}

// mergeWebSearchParams merges parameters from various sources into accumulated state.
// Follows ChatMock's _merge_from pattern.
func (s *StreamState) mergeWebSearchParams(callID string, item *WebSearchCallItem, data *WebSearchCallData) {
	if s.WebSearchState[callID] == nil {
		s.WebSearchState[callID] = &WebSearchAccum{}
	}
	accum := s.WebSearchState[callID]

	// Merge from item
	if item != nil {
		if item.Query != "" && accum.Query == "" {
			accum.Query = item.Query
		}
		if item.Parameters != nil {
			if item.Parameters.Query != "" && accum.Query == "" {
				accum.Query = item.Parameters.Query
			}
			if len(item.Parameters.Domains) > 0 && len(accum.Domains) == 0 {
				accum.Domains = item.Parameters.Domains
			}
			if item.Parameters.MaxResults > 0 && accum.MaxResults == 0 {
				accum.MaxResults = item.Parameters.MaxResults
			}
			if item.Parameters.Recency != "" && accum.Recency == "" {
				accum.Recency = item.Parameters.Recency
			}
		}
	}

	// Merge from event data
	if data != nil {
		if data.Query != "" && accum.Query == "" {
			accum.Query = data.Query
		}
		if data.Params != nil {
			if data.Params.Query != "" && accum.Query == "" {
				accum.Query = data.Params.Query
			}
			if len(data.Params.Domains) > 0 && len(accum.Domains) == 0 {
				accum.Domains = data.Params.Domains
			}
			if data.Params.MaxResults > 0 && accum.MaxResults == 0 {
				accum.MaxResults = data.Params.MaxResults
			}
			if data.Params.Recency != "" && accum.Recency == "" {
				accum.Recency = data.Params.Recency
			}
		}
	}
}

// serializeWebSearchArgs serializes accumulated web search params to JSON string.
func (s *StreamState) serializeWebSearchArgs(callID string) string {
	accum := s.WebSearchState[callID]
	if accum == nil {
		return "{}"
	}
	bytes, err := json.Marshal(accum)
	if err != nil {
		return "{}"
	}
	return string(bytes)
}

// ProcessEvent processes an SSE event and returns OpenAI chunks if applicable.
// May return multiple chunks for complex events.
func (s *StreamState) ProcessEvent(event *sse.Event) ([]*api.ChatCompletionChunk, error) {
	switch event.Event {
	case EventResponseCreated:
		var data ResponseCreatedData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}
		s.ResponseID = data.Response.ID
		s.Model = data.Response.Model
		s.Created = currentTimestamp()

		// Send initial chunk with role
		return []*api.ChatCompletionChunk{{
			ID:      s.ResponseID,
			Object:  "chat.completion.chunk",
			Created: s.Created,
			Model:   s.Model,
			Choices: []api.Choice{{
				Index: 0,
				Delta: &api.Delta{Role: "assistant"},
			}},
		}}, nil

	case EventResponseOutputTextDelta:
		var data TextDeltaData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}

		var chunks []*api.ChatCompletionChunk

		// Close think tag if open and we're getting output
		if s.ReasoningCompat == "think-tags" && s.ThinkTagOpen && !s.ThinkTagClosed {
			chunks = append(chunks, &api.ChatCompletionChunk{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index: 0,
					Delta: &api.Delta{Content: "</think>"},
				}},
			})
			s.ThinkTagOpen = false
			s.ThinkTagClosed = true
		}

		s.SawOutput = true
		s.CurrentContent += data.Delta

		chunks = append(chunks, &api.ChatCompletionChunk{
			ID:      s.ResponseID,
			Object:  "chat.completion.chunk",
			Created: s.Created,
			Model:   s.Model,
			Choices: []api.Choice{{
				Index: 0,
				Delta: &api.Delta{Content: data.Delta},
			}},
		})

		return chunks, nil

	case EventResponseOutputTextDone:
		// Text completion marker - send stop if not already sent
		if !s.SentStopChunk {
			s.SentStopChunk = true
			return []*api.ChatCompletionChunk{{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index:        0,
					Delta:        &api.Delta{},
					FinishReason: stringPtr("stop"),
				}},
			}}, nil
		}
		return nil, nil

	case EventResponseReasoningSummaryPartAdded:
		// New reasoning paragraph marker
		if s.ReasoningCompat == "think-tags" || s.ReasoningCompat == "o3" {
			if s.ReasoningSummary != "" || s.ReasoningFull != "" {
				s.PendingSummaryNewline = true
			}
		}
		return nil, nil

	case EventResponseReasoningSummaryTextDelta, EventResponseReasoningTextDelta:
		var data ReasoningTextDeltaData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}

		// Track reasoning content
		if event.Event == EventResponseReasoningSummaryTextDelta {
			s.ReasoningSummary += data.Delta
		} else {
			s.ReasoningFull += data.Delta
		}

		// Handle based on compat mode
		switch s.ReasoningCompat {
		case "none":
			// Don't emit reasoning
			return nil, nil

		case "think-tags":
			var chunks []*api.ChatCompletionChunk

			// Open think tag if not already open
			if !s.ThinkTagOpen && !s.ThinkTagClosed {
				chunks = append(chunks, &api.ChatCompletionChunk{
					ID:      s.ResponseID,
					Object:  "chat.completion.chunk",
					Created: s.Created,
					Model:   s.Model,
					Choices: []api.Choice{{
						Index: 0,
						Delta: &api.Delta{Content: "<think>"},
					}},
				})
				s.ThinkTagOpen = true
			}

			// Add newline between paragraphs
			if s.ThinkTagOpen && !s.ThinkTagClosed && s.PendingSummaryNewline {
				chunks = append(chunks, &api.ChatCompletionChunk{
					ID:      s.ResponseID,
					Object:  "chat.completion.chunk",
					Created: s.Created,
					Model:   s.Model,
					Choices: []api.Choice{{
						Index: 0,
						Delta: &api.Delta{Content: "\n"},
					}},
				})
				s.PendingSummaryNewline = false
			}

			// Emit reasoning content
			if s.ThinkTagOpen && !s.ThinkTagClosed {
				chunks = append(chunks, &api.ChatCompletionChunk{
					ID:      s.ResponseID,
					Object:  "chat.completion.chunk",
					Created: s.Created,
					Model:   s.Model,
					Choices: []api.Choice{{
						Index: 0,
						Delta: &api.Delta{Content: data.Delta},
					}},
				})
			}

			return chunks, nil

		case "o3":
			var chunks []*api.ChatCompletionChunk

			// Add newline between paragraphs
			if s.PendingSummaryNewline {
				chunks = append(chunks, &api.ChatCompletionChunk{
					ID:      s.ResponseID,
					Object:  "chat.completion.chunk",
					Created: s.Created,
					Model:   s.Model,
					Choices: []api.Choice{{
						Index: 0,
						Delta: &api.Delta{
							Reasoning: &api.ReasoningOutput{
								Content: []api.ReasoningContent{{Type: "text", Text: "\n"}},
							},
						},
					}},
				})
				s.PendingSummaryNewline = false
			}

			// Emit reasoning in o3 format
			chunks = append(chunks, &api.ChatCompletionChunk{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index: 0,
					Delta: &api.Delta{
						Reasoning: &api.ReasoningOutput{
							Content: []api.ReasoningContent{{Type: "text", Text: data.Delta}},
						},
					},
				}},
			})
			return chunks, nil

		case "legacy":
			// Emit as separate fields - only for summary events
			// Skip non-summary reasoning events to avoid empty deltas
			if event.Event != EventResponseReasoningSummaryTextDelta {
				return nil, nil
			}
			return []*api.ChatCompletionChunk{{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index: 0,
					Delta: &api.Delta{ReasoningSummary: data.Delta},
				}},
			}}, nil
		}

		return nil, nil

	case EventResponseFunctionCallArgumentsDelta:
		var data FunctionArgumentsDeltaData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}

		// Get or create tool call
		tc, ok := s.ToolCalls[data.OutputIndex]
		if !ok {
			// Tool call should have been created in output_item.added
			return nil, nil
		}
		tc.Function.Arguments += data.Delta

		return []*api.ChatCompletionChunk{{
			ID:      s.ResponseID,
			Object:  "chat.completion.chunk",
			Created: s.Created,
			Model:   s.Model,
			Choices: []api.Choice{{
				Index: 0,
				Delta: &api.Delta{
					ToolCalls: []api.ToolCall{{
						Index:    intPtr(data.OutputIndex),
						Function: api.FunctionCall{Arguments: data.Delta},
					}},
				},
			}},
		}}, nil

	case EventResponseOutputItemAdded:
		var data OutputItemAddedData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}

		// Handle any *_call type as a function tool call
		// This includes: function_call, web_search_call, mcp_call, etc.
		if strings.HasSuffix(data.Item.Type, "_call") && data.Item.Type != "message" {
			callID := data.Item.CallID
			if callID == "" {
				callID = data.Item.ID
			}

			// Determine tool name: use Name field, or derive from type (e.g., "web_search_call" -> "web_search")
			name := data.Item.Name
			if name == "" {
				name = strings.TrimSuffix(data.Item.Type, "_call")
			}

			tc := &api.ToolCall{
				ID:   callID,
				Type: "function",
				Function: api.FunctionCall{
					Name: name,
				},
			}
			s.ToolCalls[data.OutputIndex] = tc

			// Update NextToolIndex to be beyond this index to avoid conflicts
			if data.OutputIndex >= s.NextToolIndex {
				s.NextToolIndex = data.OutputIndex + 1
			}

			// Track for web search state accumulation
			if data.Item.Type == "web_search_call" {
				s.WebSearchIndex[callID] = data.OutputIndex
				s.WebSearchState[callID] = &WebSearchAccum{}
			}

			// Send initial tool call chunk
			return []*api.ChatCompletionChunk{{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index: 0,
					Delta: &api.Delta{
						ToolCalls: []api.ToolCall{{
							Index: intPtr(data.OutputIndex),
							ID:    callID,
							Type:  "function",
							Function: api.FunctionCall{
								Name: name,
							},
						}},
					},
				}},
			}}, nil
		}

		return nil, nil

	case EventResponseOutputItemDone:
		var data OutputItemDoneData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}

		// Handle any *_call type completion
		// For function_call, arguments are streamed via delta events - we just update state here
		// For web_search_call, mcp_call, etc., arguments come in this event and need to be emitted
		if strings.HasSuffix(data.Item.Type, "_call") && data.Item.Type != "message" {
			callID := data.Item.CallID
			if callID == "" {
				callID = data.Item.ID
			}

			// For function_call, arguments were already streamed via delta events
			// Just update final state, don't emit (would cause duplicate content)
			if data.Item.Type == "function_call" {
				if tc, exists := s.ToolCalls[data.OutputIndex]; exists && data.Item.Arguments != "" {
					tc.Function.Arguments = data.Item.Arguments
				}
				return nil, nil
			}

			// For other call types (web_search_call, mcp_call, etc.), emit arguments
			var argsJSON string
			if data.Item.Arguments != "" {
				argsJSON = data.Item.Arguments
			} else if data.Item.Type == "web_search_call" && data.Item.Parameters != nil {
				// web_search_call may have Parameters object instead of Arguments string
				bytes, _ := json.Marshal(data.Item.Parameters)
				argsJSON = string(bytes)
			} else if data.Item.Type == "web_search_call" {
				// Fall back to accumulated state for web_search_call
				argsJSON = s.serializeWebSearchArgs(callID)
			}

			// Find output index
			outputIndex := -1
			if idx, ok := s.WebSearchIndex[callID]; ok {
				outputIndex = idx
			} else {
				// Search in ToolCalls
				for idx, tc := range s.ToolCalls {
					if tc.ID == callID {
						outputIndex = idx
						break
					}
				}
			}

			// If we found the tool call, update and emit arguments
			// Use empty object if no arguments available (OpenAI API expects arguments field)
			if outputIndex >= 0 {
				if argsJSON == "" {
					argsJSON = "{}"
				}
				if tc, exists := s.ToolCalls[outputIndex]; exists {
					tc.Function.Arguments = argsJSON
				}

				return []*api.ChatCompletionChunk{{
					ID:      s.ResponseID,
					Object:  "chat.completion.chunk",
					Created: s.Created,
					Model:   s.Model,
					Choices: []api.Choice{{
						Index: 0,
						Delta: &api.Delta{
							ToolCalls: []api.ToolCall{{
								Index:    intPtr(outputIndex),
								Function: api.FunctionCall{Arguments: argsJSON},
							}},
						},
					}},
				}}, nil
			}

			// Tool call not tracked - this indicates the output_item.added event was missed
			slog.Debug("output_item.done for untracked tool call",
				"call_id", callID,
				"type", data.Item.Type)
		}

		return nil, nil

	case EventWebSearchCallSearching, EventWebSearchCallInProgress, EventWebSearchCallCompleted:
		var data WebSearchCallData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}

		// Get call ID from various possible locations
		callID := data.ItemID
		if callID == "" && data.Item != nil {
			callID = data.Item.CallID
			if callID == "" {
				callID = data.Item.ID
			}
		}
		if callID == "" {
			return nil, nil
		}

		// Merge params from this event into accumulated state
		s.mergeWebSearchParams(callID, data.Item, &data)

		// Get output index (may have been set in output_item.added)
		outputIndex, ok := s.WebSearchIndex[callID]
		isFirstChunk := !ok
		if !ok {
			// Not yet tracked, assign next available index
			outputIndex = s.NextToolIndex
			s.NextToolIndex++
			s.WebSearchIndex[callID] = outputIndex

			// Create tool call if not exists
			if _, exists := s.ToolCalls[outputIndex]; !exists {
				s.ToolCalls[outputIndex] = &api.ToolCall{
					ID:   callID,
					Type: "function",
					Function: api.FunctionCall{
						Name: "web_search",
					},
				}
			}
		}

		// Serialize current accumulated args
		argsJSON := s.serializeWebSearchArgs(callID)

		// Update stored tool call
		if tc, exists := s.ToolCalls[outputIndex]; exists {
			tc.Function.Arguments = argsJSON
		}

		// Emit streaming chunk with current args
		// Only include full metadata (ID, Type, Name) on first chunk; subsequent chunks only need Index + Arguments
		var toolCall api.ToolCall
		if isFirstChunk {
			toolCall = api.ToolCall{
				Index: intPtr(outputIndex),
				ID:    callID,
				Type:  "function",
				Function: api.FunctionCall{
					Name:      "web_search",
					Arguments: argsJSON,
				},
			}
		} else {
			toolCall = api.ToolCall{
				Index:    intPtr(outputIndex),
				Function: api.FunctionCall{Arguments: argsJSON},
			}
		}

		chunks := []*api.ChatCompletionChunk{{
			ID:      s.ResponseID,
			Object:  "chat.completion.chunk",
			Created: s.Created,
			Model:   s.Model,
			Choices: []api.Choice{{
				Index: 0,
				Delta: &api.Delta{
					ToolCalls: []api.ToolCall{toolCall},
				},
			}},
		}}

		// Note: We don't emit finish_reason here because there may be multiple tool calls.
		// The final finish_reason is emitted in EventResponseCompleted.

		return chunks, nil

	case EventResponseCompleted:
		var data ResponseCompletedData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}

		var chunks []*api.ChatCompletionChunk

		// Close think tag if still open
		if s.ReasoningCompat == "think-tags" && s.ThinkTagOpen && !s.ThinkTagClosed {
			chunks = append(chunks, &api.ChatCompletionChunk{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index: 0,
					Delta: &api.Delta{Content: "</think>"},
				}},
			})
			s.ThinkTagOpen = false
			s.ThinkTagClosed = true
		}

		// Determine finish reason
		finishReason := "stop"
		if len(s.ToolCalls) > 0 {
			finishReason = "tool_calls"
		}
		s.FinishReason = finishReason

		// Extract usage
		if data.Response.Usage != nil {
			s.Usage = extractUsage(data.Response.Usage)
		}

		// Send final chunk if not already sent
		if !s.SentStopChunk {
			chunks = append(chunks, &api.ChatCompletionChunk{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index:        0,
					Delta:        &api.Delta{},
					FinishReason: stringPtr(finishReason),
				}},
			})
			s.SentStopChunk = true
		}

		return chunks, nil

	case EventResponseFailed:
		var data ResponseFailedData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}
		s.FinishReason = "error"
		if data.Response.Error != nil {
			s.ErrorMessage = data.Response.Error.Message
		}
		return nil, nil

	case EventError:
		var data ErrorEventData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}
		s.FinishReason = "error"
		s.ErrorMessage = data.Message
		return nil, nil

	case EventResponseInProgress, EventResponseQueued:
		// These are status updates, no chunks to emit
		return nil, nil

	case EventResponseIncomplete:
		// Response was cut short (max tokens, content filter, etc.)
		var data ResponseIncompleteData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}

		var chunks []*api.ChatCompletionChunk

		// Close think tag if still open
		if s.ReasoningCompat == "think-tags" && s.ThinkTagOpen && !s.ThinkTagClosed {
			chunks = append(chunks, &api.ChatCompletionChunk{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index: 0,
					Delta: &api.Delta{Content: "</think>"},
				}},
			})
			s.ThinkTagOpen = false
			s.ThinkTagClosed = true
		}

		// Map incomplete reason to finish reason
		finishReason := "length" // Default for max_output_tokens
		if data.Response.IncompleteReason == "content_filter" {
			finishReason = "content_filter"
		}
		s.FinishReason = finishReason

		// Extract usage if present
		if data.Response.Usage != nil {
			s.Usage = extractUsage(data.Response.Usage)
		}

		// Send final chunk
		if !s.SentStopChunk {
			chunks = append(chunks, &api.ChatCompletionChunk{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index:        0,
					Delta:        &api.Delta{},
					FinishReason: stringPtr(finishReason),
				}},
			})
			s.SentStopChunk = true
		}

		return chunks, nil

	case EventResponseCancelled:
		s.FinishReason = "stop"
		if !s.SentStopChunk {
			s.SentStopChunk = true
			return []*api.ChatCompletionChunk{{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index:        0,
					Delta:        &api.Delta{},
					FinishReason: stringPtr("stop"),
				}},
			}}, nil
		}
		return nil, nil

	case EventResponseContentPartAdded:
		var data ContentPartAddedData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			return nil, err
		}

		// Handle refusal content parts
		if data.Part.Type == "refusal" && data.Part.Text != "" {
			s.Refusal += data.Part.Text
			return []*api.ChatCompletionChunk{{
				ID:      s.ResponseID,
				Object:  "chat.completion.chunk",
				Created: s.Created,
				Model:   s.Model,
				Choices: []api.Choice{{
					Index: 0,
					Delta: &api.Delta{Refusal: data.Part.Text},
				}},
			}}, nil
		}

		// Other content part types (output_text, etc.) are handled via their delta events
		return nil, nil

	case EventResponseContentPartDone:
		// Content part completion marker - no action needed
		return nil, nil

	case EventResponseReasoningSummaryPartDone, EventResponseReasoningTextDone,
		EventResponseReasoningSummaryTextDone, EventResponseFunctionCallArgumentsDone:
		// These are completion markers for their respective delta events
		// No additional chunks needed as the content is already streamed
		return nil, nil

	case EventFileSearchCallSearching, EventFileSearchCallInProgress, EventFileSearchCallCompleted,
		EventMCPCallInProgress, EventMCPCallCompleted, EventMCPCallFailed,
		EventMCPCallArgumentsDelta, EventMCPCallArgumentsDone,
		EventCodeInterpreterCallInProgress, EventCodeInterpreterCallInterpreting, EventCodeInterpreterCallCompleted,
		EventCodeInterpreterCallCodeDelta, EventCodeInterpreterCallCodeDone,
		EventImageGenerationCallInProgress, EventImageGenerationCallGenerating,
		EventImageGenerationCallPartialImage, EventImageGenerationCallCompleted:
		// Built-in tool events - these are server-side tools, not exposed to clients
		return nil, nil

	default:
		// Log unknown events at debug level for visibility
		slog.Debug("unknown SSE event type ignored", "event", event.Event)
		return nil, nil
	}
}

// GetUsageChunk returns a chunk with usage information for streaming.
func (s *StreamState) GetUsageChunk() *api.ChatCompletionChunk {
	if s.Usage == nil {
		return nil
	}

	// Generate system fingerprint from response ID
	systemFingerprint := ""
	if len(s.ResponseID) > 8 {
		systemFingerprint = "fp_" + s.ResponseID[len(s.ResponseID)-8:]
	}

	return &api.ChatCompletionChunk{
		ID:                s.ResponseID,
		Object:            "chat.completion.chunk",
		Created:           s.Created,
		Model:             s.Model,
		SystemFingerprint: systemFingerprint,
		Choices:           []api.Choice{}, // Empty choices array for usage-only chunk
		Usage:             s.Usage,
	}
}

// GetError returns any error message from the stream.
func (s *StreamState) GetError() string {
	return s.ErrorMessage
}

// BuildNonStreamingResponse builds a complete ChatCompletionResponse from state.
func (s *StreamState) BuildNonStreamingResponse() *api.ChatCompletionResponse {
	msg := &api.Message{
		Role: "assistant",
	}

	// Build combined reasoning text (used by multiple modes)
	reasoningText := s.ReasoningSummary
	if s.ReasoningFull != "" {
		if reasoningText != "" {
			reasoningText += "\n" // Use single newline for consistency with streaming
		}
		reasoningText += s.ReasoningFull
	}

	// Build content with reasoning based on compat mode
	content := s.CurrentContent
	switch s.ReasoningCompat {
	case "think-tags":
		if reasoningText != "" {
			content = "<think>" + reasoningText + "</think>" + content
		}
	case "o3":
		if reasoningText != "" {
			msg.Reasoning = &api.ReasoningOutput{
				Content: []api.ReasoningContent{{Type: "text", Text: reasoningText}},
			}
		}
	case "legacy":
		if s.ReasoningSummary != "" {
			msg.ReasoningSummary = s.ReasoningSummary
		}
	}
	msg.SetContentString(content)

	// Add refusal if present
	if s.Refusal != "" {
		msg.Refusal = s.Refusal
	}

	// Add tool calls if any (sorted by output index)
	// Note: For non-streaming responses, tool calls should NOT have Index field
	if len(s.ToolCalls) > 0 {
		// Find max index to iterate correctly
		maxIdx := 0
		for idx := range s.ToolCalls {
			if idx > maxIdx {
				maxIdx = idx
			}
		}
		for i := 0; i <= maxIdx; i++ {
			if tc, ok := s.ToolCalls[i]; ok {
				// Copy tool call without Index for non-streaming response
				msg.ToolCalls = append(msg.ToolCalls, api.ToolCall{
					ID:       tc.ID,
					Type:     tc.Type,
					Function: tc.Function,
				})
			}
		}
	}

	// Generate system fingerprint from response ID
	systemFingerprint := ""
	if len(s.ResponseID) > 8 {
		systemFingerprint = "fp_" + s.ResponseID[len(s.ResponseID)-8:]
	}

	return &api.ChatCompletionResponse{
		ID:                s.ResponseID,
		Object:            "chat.completion",
		Created:           s.Created,
		Model:             s.Model,
		SystemFingerprint: systemFingerprint,
		Choices: []api.Choice{{
			Index:        0,
			Message:      msg,
			FinishReason: stringPtr(s.FinishReason),
		}},
		Usage: s.Usage,
	}
}

func currentTimestamp() int64 {
	return time.Now().Unix()
}

// extractUsage converts ChatGPT usage data to OpenAI format with detailed token breakdown.
func extractUsage(usage *UsageData) *api.Usage {
	if usage == nil {
		return nil
	}

	result := &api.Usage{
		PromptTokens:     usage.InputTokens,
		CompletionTokens: usage.OutputTokens,
		TotalTokens:      usage.TotalTokens,
	}

	// Add detailed token breakdown if available
	if usage.InputTokensDetails != nil && usage.InputTokensDetails.CachedTokens > 0 {
		result.PromptTokensDetails = &api.PromptTokenDetails{
			CachedTokens: usage.InputTokensDetails.CachedTokens,
		}
	}

	if usage.OutputTokensDetails != nil && usage.OutputTokensDetails.ReasoningTokens > 0 {
		result.CompletionTokensDetails = &api.CompletionTokenDetails{
			ReasoningTokens: usage.OutputTokensDetails.ReasoningTokens,
		}
	}

	return result
}
