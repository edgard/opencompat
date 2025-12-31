// Package upstream handles communication with the ChatGPT backend API.
package chatgpt

import "encoding/json"

// ResponsesRequest represents a request to the ChatGPT Responses API.
type ResponsesRequest struct {
	Model             string           `json:"model"`
	Instructions      string           `json:"instructions"`
	Input             []InputItem      `json:"input"`
	Tools             []ToolDef        `json:"tools,omitempty"`
	ToolChoice        json.RawMessage  `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitempty"`
	Store             bool             `json:"store"`
	Stream            bool             `json:"stream"`
	Reasoning         *ReasoningConfig `json:"reasoning,omitempty"`
	Text              *TextConfig      `json:"text,omitempty"`
	Include           []string         `json:"include,omitempty"`
	PromptCacheKey    string           `json:"prompt_cache_key,omitempty"`
	// Sampling parameters
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"top_p,omitempty"`
	MaxOutputTokens *int            `json:"max_output_tokens,omitempty"`
	Stop            json.RawMessage `json:"stop,omitempty"` // string or []string
}

// InputItem represents an item in the input array.
type InputItem struct {
	Type      string          `json:"type"` // "message", "item_reference"
	Role      string          `json:"role,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"` // string or []ContentBlock
	ID        string          `json:"id,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Output    string          `json:"output,omitempty"`
	Status    string          `json:"status,omitempty"`
}

// ContentBlock represents a content block in a message.
type ContentBlock struct {
	Type     string `json:"type"` // "input_text", "input_image"
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

// ToolDef represents a tool definition for the Responses API.
type ToolDef struct {
	Type        string          `json:"type"` // "function"
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// ReasoningConfig configures reasoning behavior.
type ReasoningConfig struct {
	Effort  string `json:"effort,omitempty"`  // "none", "low", "medium", "high", "xhigh"
	Summary string `json:"summary,omitempty"` // "auto", "concise", "detailed"
}

// TextConfig configures text output.
type TextConfig struct {
	Verbosity string `json:"verbosity,omitempty"` // "low", "medium", "high"
}

// SSE Event types from ChatGPT Responses API
const (
	// Response lifecycle events
	EventResponseCreated    = "response.created"
	EventResponseInProgress = "response.in_progress"
	EventResponseCompleted  = "response.completed"
	EventResponseFailed     = "response.failed"
	EventResponseIncomplete = "response.incomplete"
	EventResponseCancelled  = "response.cancelled"
	EventResponseQueued     = "response.queued"

	// Output item events
	EventResponseOutputItemAdded = "response.output_item.added"
	EventResponseOutputItemDone  = "response.output_item.done"

	// Content part events
	EventResponseContentPartAdded = "response.content_part.added"
	EventResponseContentPartDone  = "response.content_part.done"

	// Text output events
	EventResponseOutputTextDelta = "response.output_text.delta"
	EventResponseOutputTextDone  = "response.output_text.done"

	// Function call events
	EventResponseFunctionCallArgumentsDelta = "response.function_call_arguments.delta"
	EventResponseFunctionCallArgumentsDone  = "response.function_call_arguments.done"

	// Reasoning summary events
	EventResponseReasoningSummaryPartAdded = "response.reasoning_summary_part.added"
	EventResponseReasoningSummaryPartDone  = "response.reasoning_summary_part.done"
	EventResponseReasoningSummaryTextDelta = "response.reasoning_summary_text.delta"
	EventResponseReasoningSummaryTextDone  = "response.reasoning_summary_text.done"

	// Full reasoning events
	EventResponseReasoningTextDelta = "response.reasoning_text.delta"
	EventResponseReasoningTextDone  = "response.reasoning_text.done"

	// Web search events
	EventWebSearchCallSearching  = "response.web_search_call.searching"
	EventWebSearchCallCompleted  = "response.web_search_call.completed"
	EventWebSearchCallInProgress = "response.web_search_call.in_progress"

	// File search events
	EventFileSearchCallSearching  = "response.file_search_call.searching"
	EventFileSearchCallCompleted  = "response.file_search_call.completed"
	EventFileSearchCallInProgress = "response.file_search_call.in_progress"

	// MCP (Model Context Protocol) events
	EventMCPCallInProgress     = "response.mcp_call.in_progress"
	EventMCPCallCompleted      = "response.mcp_call.completed"
	EventMCPCallFailed         = "response.mcp_call.failed"
	EventMCPCallArgumentsDelta = "response.mcp_call_arguments.delta"
	EventMCPCallArgumentsDone  = "response.mcp_call_arguments.done"

	// Code interpreter events
	EventCodeInterpreterCallInProgress   = "response.code_interpreter_call.in_progress"
	EventCodeInterpreterCallInterpreting = "response.code_interpreter_call.interpreting"
	EventCodeInterpreterCallCompleted    = "response.code_interpreter_call.completed"
	EventCodeInterpreterCallCodeDelta    = "response.code_interpreter_call_code.delta"
	EventCodeInterpreterCallCodeDone     = "response.code_interpreter_call_code.done"

	// Image generation events
	EventImageGenerationCallInProgress   = "response.image_generation_call.in_progress"
	EventImageGenerationCallGenerating   = "response.image_generation_call.generating"
	EventImageGenerationCallPartialImage = "response.image_generation_call.partial_image"
	EventImageGenerationCallCompleted    = "response.image_generation_call.completed"

	// Error event
	EventError = "error"
)

// ResponseCreatedData is the data for response.created event.
type ResponseCreatedData struct {
	Response ResponseData `json:"response"`
}

// ResponseData represents the response object.
type ResponseData struct {
	ID        string      `json:"id"`
	Object    string      `json:"object,omitempty"`
	Status    string      `json:"status"`
	Model     string      `json:"model"`
	CreatedAt int64       `json:"created_at,omitempty"`
	Error     *ErrorData  `json:"error,omitempty"`
	Metadata  interface{} `json:"metadata,omitempty"`
}

// OutputItemAddedData is the data for response.output_item.added event.
type OutputItemAddedData struct {
	OutputIndex int        `json:"output_index"`
	Item        OutputItem `json:"item"`
}

// OutputItemDoneData is the data for response.output_item.done event.
type OutputItemDoneData struct {
	OutputIndex int        `json:"output_index"`
	Item        OutputItem `json:"item"`
}

// OutputItem represents an output item.
// This struct supports all output item types from the Responses API including:
// message, function_call, web_search_call, file_search_call, mcp_call,
// code_interpreter_call, image_generation_call, computer_call, local_shell_call, etc.
type OutputItem struct {
	Type      string          `json:"type"` // "message", "function_call", "*_call" types
	ID        string          `json:"id"`
	Role      string          `json:"role,omitempty"`
	Content   []OutputContent `json:"content,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Status    string          `json:"status,omitempty"`
	// Web search fields
	Parameters *WebSearchCallParam `json:"parameters,omitempty"`
	// MCP-specific fields
	ServerLabel string     `json:"server_label,omitempty"`
	Output      string     `json:"output,omitempty"`
	Error       *ErrorData `json:"error,omitempty"`
	// Code interpreter fields
	Code        string `json:"code,omitempty"`
	ContainerID string `json:"container_id,omitempty"`
	// File search fields
	Queries []string `json:"queries,omitempty"`
	// Computer/shell action field
	Action json.RawMessage `json:"action,omitempty"`
	// Image generation field
	Result string `json:"result,omitempty"`
}

// OutputContent represents content in an output item.
type OutputContent struct {
	Type string `json:"type"` // "output_text"
	Text string `json:"text,omitempty"`
}

// TextDeltaData is the data for response.output_text.delta event.
type TextDeltaData struct {
	OutputIndex  int    `json:"output_index"`
	ContentIndex int    `json:"content_index"`
	Delta        string `json:"delta"`
}

// FunctionArgumentsDeltaData is the data for function_call_arguments.delta event.
type FunctionArgumentsDeltaData struct {
	OutputIndex int    `json:"output_index"`
	CallID      string `json:"call_id"`
	Delta       string `json:"delta"`
}

// ResponseCompletedData is the data for response.completed event.
type ResponseCompletedData struct {
	Response ResponseCompletedResponse `json:"response"`
}

// ResponseCompletedResponse contains the completed response.
type ResponseCompletedResponse struct {
	ID     string       `json:"id"`
	Status string       `json:"status"`
	Output []OutputItem `json:"output"`
	Usage  *UsageData   `json:"usage,omitempty"`
}

// UsageData contains token usage information.
type UsageData struct {
	InputTokens         int                `json:"input_tokens"`
	OutputTokens        int                `json:"output_tokens"`
	TotalTokens         int                `json:"total_tokens"`
	InputTokensDetails  *TokenDetails      `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *OutputTokenDetail `json:"output_tokens_details,omitempty"`
}

// TokenDetails contains detailed input token information.
type TokenDetails struct {
	CachedTokens    int `json:"cached_tokens,omitempty"`
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// OutputTokenDetail contains detailed output token information.
type OutputTokenDetail struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// ResponseFailedData is the data for response.failed event.
type ResponseFailedData struct {
	Response ResponseFailedResponse `json:"response"`
}

// ResponseFailedResponse contains the failed response.
type ResponseFailedResponse struct {
	ID     string     `json:"id"`
	Status string     `json:"status"`
	Error  *ErrorData `json:"error,omitempty"`
}

// ErrorData contains error information.
type ErrorData struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ErrorEventData is the data for error event.
type ErrorEventData struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ReasoningTextDeltaData is the data for reasoning text delta events.
type ReasoningTextDeltaData struct {
	OutputIndex  int    `json:"output_index"`
	ContentIndex int    `json:"content_index"`
	Delta        string `json:"delta"`
}

// WebSearchCallData is the data for web search events.
type WebSearchCallData struct {
	ItemID      string              `json:"item_id"`
	OutputIndex int                 `json:"output_index,omitempty"`
	Item        *WebSearchCallItem  `json:"item,omitempty"`
	Query       string              `json:"query,omitempty"`
	Params      *WebSearchCallParam `json:"parameters,omitempty"`
}

// WebSearchCallItem represents a web search call item.
type WebSearchCallItem struct {
	Type       string              `json:"type"` // "web_search_call"
	ID         string              `json:"id"`
	CallID     string              `json:"call_id"`
	Status     string              `json:"status"`
	Query      string              `json:"query,omitempty"`
	Parameters *WebSearchCallParam `json:"parameters,omitempty"`
}

// WebSearchCallParam represents web search parameters.
type WebSearchCallParam struct {
	Query      string   `json:"query,omitempty"`
	Domains    []string `json:"domains,omitempty"`
	MaxResults int      `json:"max_results,omitempty"`
	Recency    string   `json:"recency,omitempty"`
}

// ContentPartAddedData is the data for response.content_part.added event.
type ContentPartAddedData struct {
	OutputIndex  int         `json:"output_index"`
	ContentIndex int         `json:"content_index"`
	Part         ContentPart `json:"part"`
}

// ContentPartDoneData is the data for response.content_part.done event.
type ContentPartDoneData struct {
	OutputIndex  int         `json:"output_index"`
	ContentIndex int         `json:"content_index"`
	Part         ContentPart `json:"part"`
}

// ContentPart represents a content part in an output item.
type ContentPart struct {
	Type string `json:"type"` // "output_text", "refusal", etc.
	Text string `json:"text,omitempty"`
}

// ResponseIncompleteData is the data for response.incomplete event.
type ResponseIncompleteData struct {
	Response ResponseIncompleteResponse `json:"response"`
}

// ResponseIncompleteResponse contains the incomplete response details.
type ResponseIncompleteResponse struct {
	ID               string       `json:"id"`
	Status           string       `json:"status"`
	Output           []OutputItem `json:"output,omitempty"`
	Usage            *UsageData   `json:"usage,omitempty"`
	IncompleteReason string       `json:"incomplete_reason,omitempty"` // "max_output_tokens", "content_filter"
}

// FunctionCallArgumentsDoneData is the data for function_call_arguments.done event.
type FunctionCallArgumentsDoneData struct {
	OutputIndex int    `json:"output_index"`
	CallID      string `json:"call_id"`
	Arguments   string `json:"arguments"`
}
