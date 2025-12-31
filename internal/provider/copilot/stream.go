package copilot

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/edgard/opencompat/internal/api"
	"github.com/edgard/opencompat/internal/sse"
)

// Stream implements the provider.Stream interface for Copilot responses.
// Copilot uses standard OpenAI format, so this is a thin pass-through wrapper.
type Stream struct {
	resp          *http.Response
	reader        *sse.Reader
	streaming     bool
	done          bool
	statusChecked bool
	response      *api.ChatCompletionResponse
	err           error
}

// NewStream creates a new stream from an HTTP response.
func NewStream(resp *http.Response, streaming bool) *Stream {
	s := &Stream{
		resp:      resp,
		streaming: streaming,
	}
	if streaming {
		s.reader = sse.NewReader(resp.Body)
	}
	return s
}

// Next returns the next chunk from the stream.
// For non-streaming requests, returns io.EOF immediately (use Response() to get the result).
func (s *Stream) Next() (*api.ChatCompletionChunk, error) {
	if s.done {
		return nil, io.EOF
	}

	// Check HTTP status once
	if !s.statusChecked {
		s.statusChecked = true
		if s.resp.StatusCode != http.StatusOK {
			s.done = true
			body, _ := io.ReadAll(s.resp.Body)
			s.err = api.NewUpstreamError(s.resp.StatusCode, parseUpstreamError(body))
			return nil, s.err
		}

		// For non-streaming: read response immediately and return EOF
		if !s.streaming {
			s.done = true
			return nil, s.readNonStreaming()
		}
	}

	// Streaming: read next SSE event
	for {
		event, err := s.reader.ReadEvent()
		if err != nil {
			s.done = true
			if err != io.EOF {
				s.err = err
			}
			return nil, err
		}

		// Skip empty events
		if len(event.Data) == 0 {
			continue
		}

		// Parse chunk
		var chunk api.ChatCompletionChunk
		if err := json.Unmarshal(event.Data, &chunk); err != nil {
			continue // Skip malformed events
		}

		normalizeChunk(&chunk)
		return &chunk, nil
	}
}

// readNonStreaming reads and parses a non-streaming response.
// Returns io.EOF on success (response available via Response()), or error on failure.
func (s *Stream) readNonStreaming() error {
	body, err := io.ReadAll(s.resp.Body)
	if err != nil {
		s.err = err
		return err
	}

	var resp api.ChatCompletionResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		s.err = err
		return err
	}

	normalizeResponse(&resp)
	s.response = &resp
	return io.EOF
}

// Response returns the non-streaming response.
func (s *Stream) Response() *api.ChatCompletionResponse {
	return s.response
}

// Err returns any error that occurred during streaming.
func (s *Stream) Err() error {
	return s.err
}

// Close releases resources associated with the stream.
func (s *Stream) Close() error {
	if s.resp != nil && s.resp.Body != nil {
		return s.resp.Body.Close()
	}
	return nil
}

// normalizeChunk ensures OpenAI-required fields are set on streaming chunks.
func normalizeChunk(chunk *api.ChatCompletionChunk) {
	if chunk.Object == "" {
		chunk.Object = "chat.completion.chunk"
	}
	if chunk.Created == 0 {
		chunk.Created = time.Now().Unix()
	}
}

// normalizeResponse ensures OpenAI-required fields are set on non-streaming responses.
func normalizeResponse(resp *api.ChatCompletionResponse) {
	if resp.Object == "" {
		resp.Object = "chat.completion"
	}
	if resp.Created == 0 {
		resp.Created = time.Now().Unix()
	}
}

// parseUpstreamError extracts a meaningful error message from upstream response.
func parseUpstreamError(body []byte) string {
	var errResp struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		Message string `json:"message"`
	}

	message := ""
	if err := json.Unmarshal(body, &errResp); err == nil {
		if errResp.Error.Message != "" {
			message = errResp.Error.Message
		} else if errResp.Message != "" {
			message = errResp.Message
		}
	}

	if message == "" {
		bodyStr := string(body)
		if len(bodyStr) > 500 {
			bodyStr = bodyStr[:500] + "..."
		}
		if bodyStr == "" {
			return "unknown error"
		}
		message = bodyStr
	}

	// Enhance error messages with helpful hints
	return enhanceErrorMessage(message)
}

// enhanceErrorMessage adds helpful context to known error messages.
func enhanceErrorMessage(message string) string {
	lower := strings.ToLower(message)
	// Help users when a model isn't enabled in their Copilot settings
	// Check for "model" to avoid matching unrelated "not supported" errors
	if strings.Contains(lower, "model") &&
		(strings.Contains(lower, "not supported") || strings.Contains(lower, "not available")) {
		return message + "\n\nMake sure the model is enabled in your Copilot settings: https://github.com/settings/copilot"
	}
	return message
}
