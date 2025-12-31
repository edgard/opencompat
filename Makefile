# OpenCompat Makefile

# Binary name
BINARY := opencompat

# Build variables
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DATE    ?= $(shell date -u '+%Y-%m-%d_%H:%M:%S')
LDFLAGS := -ldflags "-s -w -X main.version=$(VERSION) -X main.commit=$(COMMIT) -X main.date=$(DATE)"

# Go variables
GOBIN   := $(shell go env GOPATH)/bin
GOOS    ?= $(shell go env GOOS)
GOARCH  ?= $(shell go env GOARCH)

# Directories
BUILD_DIR := build
DIST_DIR  := dist

.PHONY: all build install clean test lint fmt vet check run help
.PHONY: build-all release dev tidy update
.PHONY: test-e2e test-e2e-chatgpt test-e2e-copilot test-e2e-all

# Default target
all: check build

## Build targets

build: ## Build the binary for current platform
	@echo "Building $(BINARY)..."
	@go build $(LDFLAGS) -o $(BINARY) .

build-all: clean ## Build binaries for all platforms
	@echo "Building for all platforms..."
	@mkdir -p $(DIST_DIR)
	@GOOS=linux   GOARCH=amd64 go build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY)-linux-amd64 .
	@GOOS=linux   GOARCH=arm64 go build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY)-linux-arm64 .
	@GOOS=darwin  GOARCH=amd64 go build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY)-darwin-amd64 .
	@GOOS=darwin  GOARCH=arm64 go build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY)-darwin-arm64 .
	@GOOS=windows GOARCH=amd64 go build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY)-windows-amd64.exe .
	@echo "Binaries built in $(DIST_DIR)/"

install: build ## Install the binary to GOPATH/bin
	@echo "Installing $(BINARY) to $(GOBIN)..."
	@cp $(BINARY) $(GOBIN)/$(BINARY)

uninstall: ## Remove the binary from GOPATH/bin
	@echo "Removing $(BINARY) from $(GOBIN)..."
	@rm -f $(GOBIN)/$(BINARY)

## Development targets

run: build ## Build and run the server
	@./$(BINARY) serve

dev: ## Run with go run (faster iteration)
	@go run . serve

## Quality targets

test: ## Run tests
	@echo "Running tests..."
	@go test -v -race -cover ./...

test-short: ## Run tests (short mode)
	@go test -short ./...

test-e2e: test-e2e-all ## Run E2E tests for all providers

test-e2e-chatgpt: ## Run E2E tests for ChatGPT provider
	@echo "Running E2E tests (ChatGPT provider)..."
	@cd tests && \
		if [ ! -d .venv ]; then python3 -m venv .venv; fi && \
		. .venv/bin/activate && \
		pip install -q -r requirements.txt && \
		python e2e.py --provider chatgpt

test-e2e-copilot: ## Run E2E tests for Copilot provider
	@echo "Running E2E tests (Copilot provider)..."
	@cd tests && \
		if [ ! -d .venv ]; then python3 -m venv .venv; fi && \
		. .venv/bin/activate && \
		pip install -q -r requirements.txt && \
		python e2e.py --provider copilot

test-e2e-all: ## Run E2E tests for all providers
	@echo "Running E2E tests (all providers)..."
	@cd tests && \
		if [ ! -d .venv ]; then python3 -m venv .venv; fi && \
		. .venv/bin/activate && \
		pip install -q -r requirements.txt && \
		echo "=== ChatGPT Provider ===" && \
		python e2e.py --provider chatgpt && \
		echo "" && \
		echo "=== Copilot Provider ===" && \
		python e2e.py --provider copilot

coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	@go test -race -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

bench: ## Run benchmarks
	@echo "Running benchmarks..."
	@go test -bench=. -benchmem ./...

lint: ## Run linter (requires golangci-lint)
	@echo "Tidying dependencies..."
	@go mod tidy
	@echo "Running linter..."
	@golangci-lint run ./...

fmt: ## Format code
	@echo "Formatting code..."
	@go fmt ./...
	@gofmt -s -w .

vet: ## Run go vet
	@echo "Running go vet..."
	@go vet ./...

check: fmt vet lint ## Run all checks (fmt, vet, lint)
	@echo "All checks passed!"

## Dependency targets

tidy: ## Tidy and verify dependencies
	@echo "Tidying dependencies..."
	@go mod tidy
	@go mod verify

update: ## Update dependencies
	@echo "Updating dependencies..."
	@go get -u ./...
	@go mod tidy

## Cleanup targets

clean: ## Remove build artifacts
	@echo "Cleaning..."
	@rm -f $(BINARY)
	@rm -rf $(BUILD_DIR) $(DIST_DIR)
	@rm -f coverage.out coverage.html

clean-all: clean ## Remove build artifacts and cached data
	@echo "Cleaning cached data..."
	@rm -rf ~/.local/share/opencompat
	@rm -rf ~/.cache/opencompat

## Release targets

release: clean build-all ## Create release archives
	@echo "Creating release archives..."
	@cd $(DIST_DIR) && \
		for f in $(BINARY)-*; do \
			if [ -f "$$f" ]; then \
				if echo "$$f" | grep -q "windows"; then \
					zip "$${f%.exe}.zip" "$$f"; \
				else \
					tar -czf "$$f.tar.gz" "$$f"; \
				fi; \
			fi; \
		done
	@echo "Release archives created in $(DIST_DIR)/"

## Information targets

version: ## Show version information
	@echo "Version: $(VERSION)"
	@echo "Commit:  $(COMMIT)"
	@echo "Date:    $(DATE)"
	@echo "OS/Arch: $(GOOS)/$(GOARCH)"

info: ## Show project information
	@echo "Binary:     $(BINARY)"
	@echo "Go version: $(shell go version)"
	@echo "GOPATH:     $(shell go env GOPATH)"
	@echo "GOBIN:      $(GOBIN)"

## Help target

help: ## Show this help
	@echo "OpenCompat - Personal API compatibility layer"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:[^#]*## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":[^#]*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
