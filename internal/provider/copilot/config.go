package copilot

import (
	"os"
	"strconv"

	"github.com/edgard/opencompat/internal/auth"
)

// Provider identification
const ProviderID = "copilot"

// Environment variable names for Copilot provider
const (
	EnvModelsRefresh = "OPENCOMPAT_COPILOT_MODELS_REFRESH"
)

// Default values
const (
	DefaultModelsRefresh = 24 * 60 // 24 hours in minutes
)

// OAuth Device Flow configuration for GitHub
const (
	GitHubClientID       = "Iv1.b507a08c87ecfe98"
	GitHubDeviceCodeURL  = "https://github.com/login/device/code"
	GitHubAccessTokenURL = "https://github.com/login/oauth/access_token"
	GitHubScopes         = "read:user"
)

// Copilot API configuration
const (
	CopilotTokenURL = "https://api.github.com/copilot_internal/v2/token"
	CopilotBaseURL  = "https://api.githubcopilot.com"
	CopilotChatURL  = CopilotBaseURL + "/chat/completions"
)

// Required headers for Copilot API
const (
	EditorVersion        = "vscode/1.95.0"
	EditorPluginVersion  = "copilot-chat/0.26.7"
	CopilotIntegrationID = "vscode-chat"
	UserAgentValue       = "GitHubCopilotChat/0.26.7"
	GitHubAPIVersion     = "2025-04-01"
)

// Config holds Copilot-specific configuration.
type Config struct {
	ModelsRefresh int // refresh interval in minutes
}

// LoadConfig reads Copilot configuration from environment variables.
func LoadConfig() *Config {
	return &Config{
		ModelsRefresh: getEnvInt(EnvModelsRefresh, DefaultModelsRefresh),
	}
}

// EnvVarDoc documents an environment variable.
type EnvVarDoc struct {
	Name        string
	Description string
	Default     string
}

// EnvVarDocs returns documentation for environment variables.
func EnvVarDocs() []EnvVarDoc {
	return []EnvVarDoc{
		{Name: EnvModelsRefresh, Description: "Models refresh interval in minutes", Default: strconv.Itoa(DefaultModelsRefresh)},
	}
}

func getEnvInt(key string, defaultVal int) int {
	if val := os.Getenv(key); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			return i
		}
	}
	return defaultVal
}

// GetDeviceFlowConfig returns the device flow configuration for GitHub Copilot.
func GetDeviceFlowConfig() *auth.DeviceFlowConfig {
	return &auth.DeviceFlowConfig{
		ClientID:       GitHubClientID,
		Scopes:         GitHubScopes,
		DeviceCodeURL:  GitHubDeviceCodeURL,
		AccessTokenURL: GitHubAccessTokenURL,
		UserAgent:      UserAgentValue,
	}
}
