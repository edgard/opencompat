# OpenCompat

> **NOTICE**: This is an independent open-source project for **personal,
> non-commercial use only**. It is NOT affiliated with, endorsed by, or
> sponsored by OpenAI, GitHub, Microsoft, or any other company. Users are
> responsible for compliance with all applicable terms of service. See
> [Disclaimer](#disclaimer).

A personal API compatibility layer that provides an OpenAI-compatible interface
for your existing subscriptions. For individual, non-commercial use only.

## Overview

OpenCompat provides a local API server with endpoints compatible with standard
AI client libraries (`/v1/chat/completions`, `/v1/models`). It allows you to
use your existing subscriptions through tools that support the OpenAI API format.

### Features

- OpenAI-compatible API endpoints
- Multi-provider architecture (ChatGPT and GitHub Copilot)
- OAuth authentication with PKCE (ChatGPT)
- GitHub device flow authentication (Copilot)
- Automatic token refresh
- Streaming and non-streaming responses
- Tool/function calling support
- Image input support

## Installation

### From Source

```bash
git clone https://github.com/edgard/opencompat.git
cd opencompat
make build
```

### Pre-built Binaries

Download from the [Releases](https://github.com/edgard/opencompat/releases) page.

### Docker

```bash
docker pull ghcr.io/edgard/opencompat:latest
```

## Quick Start

```bash
# 1. Login with your account (choose one or both)
opencompat login chatgpt   # Opens browser for OAuth
opencompat login copilot   # Uses GitHub device flow

# 2. Start the server
opencompat serve

# 3. Use with any OpenAI-compatible client
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatgpt/gpt-5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Docker

The Docker image runs in server mode only. You must authenticate on your host
machine first, then mount the credentials directory:

```bash
# 1. Login on host (credentials stored in $XDG_DATA_HOME/opencompat or ~/.local/share/opencompat)
opencompat login chatgpt

# 2. Run container with mounted credentials
docker run -p 8080:8080 \
  -v ~/.local/share/opencompat:/home/nonroot/.local/share/opencompat:ro \
  ghcr.io/edgard/opencompat:latest
```

With environment variables:

```bash
docker run -p 8080:8080 \
  -v ~/.local/share/opencompat:/home/nonroot/.local/share/opencompat:ro \
  -e OPENCOMPAT_LOG_LEVEL=debug \
  ghcr.io/edgard/opencompat:latest
```

## Usage

### Commands

```bash
opencompat login <provider>   # Authenticate with a provider (opens browser)
opencompat logout <provider>  # Remove stored credentials for a provider
opencompat info               # Show authentication status for all providers
opencompat models             # List all supported providers and models
opencompat serve              # Start the API server (default)
opencompat version            # Show version information
opencompat help               # Show help message
```

### Providers

| Provider | Auth Method | Description |
|----------|-------------|-------------|
| `chatgpt` | OAuth (browser) | ChatGPT with Codex models |
| `copilot` | GitHub device flow | GitHub Copilot models |

### Parameter Support

Not all parameters are supported by all providers. The table below shows which
parameters are supported (passed to upstream API) vs ignored (accepted but not used).

| Parameter | ChatGPT | Copilot |
|-----------|---------|---------|
| `temperature` | Supported | Supported |
| `top_p` | Supported | Supported |
| `max_tokens` | Supported | Supported |
| `max_completion_tokens` | Supported | Supported |
| `stop` | Supported | Supported |
| `presence_penalty` | Ignored | Supported |
| `frequency_penalty` | Ignored | Supported |
| `response_format` | Ignored | Supported |
| `parallel_tool_calls` | Supported | Supported |
| `reasoning_effort` | Supported | Ignored |
| `n` | Ignored | Ignored |
| `seed` | Ignored | Ignored |
| `logit_bias` | Ignored | Ignored |
| `user` | Ignored | Ignored |

Note: "Ignored" means the parameter is accepted without error but has no effect.
This ensures compatibility with clients that send these parameters.

### Model Format

Models must be prefixed with the provider name:

#### ChatGPT Models

```
chatgpt/gpt-5.2-codex
chatgpt/gpt-5.1-codex-max
chatgpt/gpt-5.1-codex
chatgpt/gpt-5-codex
chatgpt/gpt-5.1-codex-mini
chatgpt/gpt-5.2
chatgpt/gpt-5.1
chatgpt/gpt-5
```

#### Copilot Models

Copilot models are fetched dynamically from the API. Use `opencompat models` to list available models.

#### Effort Suffixes (ChatGPT only)

ChatGPT models can include an effort suffix to control reasoning effort:

```
chatgpt/gpt-5.1-codex-low
chatgpt/gpt-5.1-codex-medium
chatgpt/gpt-5.1-codex-high
```

Alternatively, set reasoning effort via the `reasoning_effort` parameter in the request body.

Use `opencompat models` to list all available models.

### Environment Variables

#### Global

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCOMPAT_HOST` | `127.0.0.1` | Server bind address |
| `OPENCOMPAT_PORT` | `8080` | Server listen port |
| `OPENCOMPAT_LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |
| `OPENCOMPAT_LOG_FORMAT` | `text` | Log format (text, json) |

#### ChatGPT Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCOMPAT_CHATGPT_INSTRUCTIONS_REFRESH` | `1440` | Instructions refresh interval (minutes) |

#### Copilot Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCOMPAT_COPILOT_MODELS_REFRESH` | `1440` | Models refresh interval (minutes) |

### Per-Request Headers (ChatGPT only)

The following HTTP headers configure ChatGPT provider behavior on a per-request basis:

| Header | Default | Values |
|--------|---------|--------|
| `X-Reasoning-Summary` | `auto` | auto, concise, detailed |
| `X-Reasoning-Compat` | `none` | none, think-tags, o3, legacy |
| `X-Text-Verbosity` | `medium` | low, medium, high |

#### Reasoning Compat Modes

The `X-Reasoning-Compat` header controls how reasoning/thinking content is included in responses:

| Mode | Description |
|------|-------------|
| `none` | No reasoning content included in responses (default) |
| `think-tags` | Reasoning wrapped in `<think>...</think>` tags, prepended to content |
| `o3` | Reasoning in separate `reasoning` field with structured content |
| `legacy` | Reasoning summary in `reasoning_summary` field (summary only, not full reasoning) |

Note: Reasoning effort can also be set via model suffix (see [Model Format](#model-format)) or the `reasoning_effort` request parameter.

Example:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Reasoning-Compat: think-tags" \
  -d '{
    "model": "chatgpt/gpt-5.1-codex-high",
    "messages": [{"role": "user", "content": "Solve this step by step"}]
  }'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## Client Examples

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="chatgpt/gpt-5",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Node.js

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://127.0.0.1:8080/v1',
  apiKey: 'not-needed',
});

const response = await client.chat.completions.create({
  model: 'chatgpt/gpt-5',
  messages: [{ role: 'user', content: 'Hello!' }],
});

console.log(response.choices[0].message.content);
```

### cURL

```bash
# Streaming request with reasoning
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Reasoning-Compat: think-tags" \
  -d '{
    "model": "chatgpt/gpt-5.1-codex-high",
    "messages": [{"role": "user", "content": "Explain recursion"}],
    "stream": true
  }'
```

## Development

```bash
make build      # Build the binary
make check      # Run all checks (fmt, vet, lint)
make test       # Run tests
make dev        # Run with go run (faster iteration)
make build-all  # Build for all platforms
make test-e2e   # Run E2E tests (requires server running)
make help       # Show all available targets
```

## Requirements

- Go 1.21+ (for building from source)
- A compatible subscription (ChatGPT Plus/Pro or GitHub Copilot)
- A web browser for ChatGPT OAuth login (Copilot uses device flow)

## Technical Notes

This software:
- Uses standard OAuth PKCE authentication (ChatGPT)
- Uses GitHub device flow authentication (Copilot)
- Translates between API formats
- Uses your own credentials and subscription
- Fetches instruction files from open-source repositories (Apache 2.0)

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Disclaimer

### No Affiliation

This software is NOT affiliated with, endorsed by, or sponsored by OpenAI,
GitHub, Microsoft, or any other company. This is an independent open-source
project.

### Personal, Non-Commercial Use Only

This software is for personal, individual, non-commercial use only.

Not intended for:
- Commercial services or resale
- Multi-user or shared deployments
- High-volume automated processing

### Your Responsibility

You are responsible for:
- Compliance with all applicable terms of service
- How you use this software
- Any consequences of your use

### Assumption of Risk

By using this software, you acknowledge:
- Your use may be subject to third-party terms of service
- You assume all risk for any consequences
- The author is not liable for any damages

### No Warranty

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. See LICENSE
for full terms.
