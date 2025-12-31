// Package copilot implements the GitHub Copilot provider.
package copilot

import (
	"context"

	"github.com/edgard/opencompat/internal/api"
	"github.com/edgard/opencompat/internal/auth"
	"github.com/edgard/opencompat/internal/provider"
)

func init() {
	provider.AddRegistration(func(r *provider.Registry) {
		r.RegisterMeta(provider.ProviderMeta{
			ID:            ProviderID,
			Name:          "GitHub Copilot",
			AuthMethod:    auth.AuthMethodDeviceFlow,
			DeviceFlowCfg: GetDeviceFlowConfig(),
			EnvVars:       convertEnvVarDocs(EnvVarDocs()),
			Factory:       New,
		})
	})
}

// convertEnvVarDocs converts copilot.EnvVarDoc to provider.EnvVarDoc.
func convertEnvVarDocs(docs []EnvVarDoc) []provider.EnvVarDoc {
	result := make([]provider.EnvVarDoc, len(docs))
	for i, d := range docs {
		result[i] = provider.EnvVarDoc{
			Name:        d.Name,
			Description: d.Description,
			Default:     d.Default,
		}
	}
	return result
}

// Provider implements the Copilot provider.
type Provider struct {
	client      *Client
	modelsCache *ModelsCache
	cfg         *Config
}

// New creates a new Copilot provider.
func New(store *auth.Store) (provider.Provider, error) {
	cfg := LoadConfig()
	client := NewClient(store)
	return &Provider{
		client:      client,
		modelsCache: NewModelsCache(client, cfg.ModelsRefresh),
		cfg:         cfg,
	}, nil
}

// ID returns the provider identifier.
func (p *Provider) ID() string {
	return ProviderID
}

// Models returns the list of supported models.
func (p *Provider) Models() []api.Model {
	return p.modelsCache.GetModels()
}

// SupportsModel checks if a model ID is supported.
func (p *Provider) SupportsModel(modelID string) bool {
	return p.modelsCache.SupportsModel(modelID)
}

// ChatCompletion sends a chat completion request.
func (p *Provider) ChatCompletion(ctx context.Context, req *provider.ChatCompletionRequest) (provider.Stream, error) {
	// Transform messages: convert system role to assistant (Copilot compatibility)
	messages := transformMessages(req.Messages)

	// Convert provider request to API request for Copilot
	chatReq := &api.ChatCompletionRequest{
		Model:               req.Model,
		Messages:            messages,
		Tools:               req.Tools,
		ToolChoice:          req.ToolChoice,
		Stream:              req.Stream,
		StreamOptions:       req.StreamOptions,
		Temperature:         req.Temperature,
		TopP:                req.TopP,
		MaxTokens:           req.MaxTokens,
		MaxCompletionTokens: req.MaxCompletionTokens,
		Stop:                req.Stop,
		PresencePenalty:     req.PresencePenalty,
		FrequencyPenalty:    req.FrequencyPenalty,
		ResponseFormat:      req.ResponseFormat,
		ParallelToolCalls:   req.ParallelToolCalls,
	}

	// Send request
	resp, err := p.client.SendRequest(ctx, chatReq)
	if err != nil {
		return nil, err
	}

	return NewStream(resp, req.Stream), nil
}

// transformMessages converts system messages to assistant role for Copilot compatibility.
func transformMessages(messages []api.Message) []api.Message {
	result := make([]api.Message, len(messages))
	for i, msg := range messages {
		result[i] = msg
		if msg.Role == "system" {
			result[i].Role = "assistant"
		}
	}
	return result
}

// Init performs initialization - fetches models list.
func (p *Provider) Init() error {
	// Trigger initial models fetch
	_ = p.modelsCache.GetModels()
	return nil
}

// Start begins background tasks.
func (p *Provider) Start() {
	p.modelsCache.StartBackgroundRefresh()
}

// Close stops background tasks.
func (p *Provider) Close() {
	p.modelsCache.StopBackgroundRefresh()
}

// RefreshModels forces a refresh of the models list.
func (p *Provider) RefreshModels(ctx context.Context) error {
	return p.modelsCache.RefreshModels(ctx)
}
