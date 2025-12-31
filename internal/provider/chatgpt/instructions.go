package chatgpt

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// InstructionsCache manages caching of Codex instructions from GitHub.
type InstructionsCache struct {
	mu              sync.RWMutex
	cache           map[string]*cacheEntry
	version         string
	refreshInterval time.Duration
}

type cacheEntry struct {
	content   string
	fetchedAt time.Time
}

type cacheMeta struct {
	Version   string    `json:"version"`
	FetchedAt time.Time `json:"fetched_at"`
	ETag      string    `json:"etag,omitempty"`
}

// NewInstructionsCache creates a new instructions cache.
func NewInstructionsCache() *InstructionsCache {
	return &InstructionsCache{
		cache:           make(map[string]*cacheEntry),
		refreshInterval: time.Duration(DefaultInstructionsRefresh) * time.Minute,
	}
}

// SetRefreshInterval sets the memory cache refresh interval.
func (c *InstructionsCache) SetRefreshInterval(interval time.Duration) {
	c.mu.Lock()
	c.refreshInterval = interval
	c.mu.Unlock()
}

// Prefetch fetches all prompt files on startup.
// Returns error if any file cannot be fetched AND has no valid disk cache.
func (c *InstructionsCache) Prefetch() error {
	promptFiles := GetAllPromptFiles()
	var errs []string

	slog.Debug("prefetching instruction files", "count", len(promptFiles))

	for _, promptFile := range promptFiles {
		content, err := c.prefetchOne(promptFile)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", promptFile, err))
			continue
		}

		c.mu.Lock()
		c.cache[promptFile] = &cacheEntry{
			content:   content,
			fetchedAt: time.Now(),
		}
		c.mu.Unlock()

		slog.Debug("loaded instruction file", "file", promptFile)
	}

	if len(errs) > 0 {
		return fmt.Errorf("failed to prefetch instructions:\n  %v", errs)
	}

	slog.Debug("all instruction files loaded successfully")
	return nil
}

// prefetchOne fetches a single prompt file, trying GitHub first, then disk cache.
func (c *InstructionsCache) prefetchOne(promptFile string) (string, error) {
	// Try GitHub first
	content, err := c.fetchFromGitHub(promptFile)
	if err == nil {
		// Save to disk cache (async)
		go func(pf, content string) {
			if err := c.saveToDisk(pf, content); err != nil {
				slog.Warn("failed to save instruction to disk cache",
					"file", pf,
					"error", err,
				)
			}
		}(promptFile, content)
		return content, nil
	}

	slog.Warn("github fetch failed, trying disk cache",
		"file", promptFile,
		"error", err,
	)

	// Fallback to disk cache (even if expired)
	content, diskErr := c.loadFromDiskWithExpired(promptFile)
	if diskErr == nil {
		return content, nil
	}

	return "", fmt.Errorf("github: %w, disk cache: %v", err, diskErr)
}

// StartBackgroundRefresh starts a goroutine that periodically refreshes all instructions.
func (c *InstructionsCache) StartBackgroundRefresh(ctx context.Context, interval time.Duration) {
	c.SetRefreshInterval(interval)

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				slog.Debug("background instructions refresh stopped")
				return
			case <-ticker.C:
				c.refreshAll()
			}
		}
	}()

	slog.Debug("background instructions refresh started", "interval", interval)
}

// refreshAll refreshes all prompt files in the background.
func (c *InstructionsCache) refreshAll() {
	promptFiles := GetAllPromptFiles()
	slog.Debug("background refresh started", "count", len(promptFiles))

	successCount := 0
	for _, promptFile := range promptFiles {
		content, err := c.fetchFromGitHub(promptFile)
		if err != nil {
			slog.Warn("failed to refresh instruction file",
				"file", promptFile,
				"error", err,
			)
			continue
		}

		c.mu.Lock()
		c.cache[promptFile] = &cacheEntry{
			content:   content,
			fetchedAt: time.Now(),
		}
		c.mu.Unlock()

		// Save to disk cache (async)
		go func(pf, content string) {
			if err := c.saveToDisk(pf, content); err != nil {
				slog.Warn("failed to save instruction to disk cache",
					"file", pf,
					"error", err,
				)
			}
		}(promptFile, content)
		successCount++
	}

	slog.Debug("background refresh complete",
		"success", successCount,
		"total", len(promptFiles),
	)
}

// RefreshAll forces a refresh of all instruction files.
// Returns error if any file cannot be fetched.
func (c *InstructionsCache) RefreshAll(ctx context.Context) error {
	promptFiles := GetAllPromptFiles()
	slog.Debug("force refreshing instruction files", "count", len(promptFiles))

	var errs []string
	for _, promptFile := range promptFiles {
		// Check context
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		content, err := c.fetchFromGitHub(promptFile)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", promptFile, err))
			continue
		}

		c.mu.Lock()
		c.cache[promptFile] = &cacheEntry{
			content:   content,
			fetchedAt: time.Now(),
		}
		c.mu.Unlock()

		// Save to disk cache (async)
		go func(pf, content string) {
			if err := c.saveToDisk(pf, content); err != nil {
				slog.Warn("failed to save instruction to disk cache",
					"file", pf,
					"error", err,
				)
			}
		}(promptFile, content)
	}

	if len(errs) > 0 {
		return fmt.Errorf("failed to refresh some instructions:\n  %v", errs)
	}

	slog.Debug("all instruction files refreshed successfully")
	return nil
}

// Get retrieves instructions for a model from cache.
// After prefetch, this should always return from memory cache.
func (c *InstructionsCache) Get(modelID string) (string, error) {
	promptFile := GetPromptFile(modelID)

	// Check memory cache first
	c.mu.RLock()
	entry, ok := c.cache[promptFile]
	refreshInterval := c.refreshInterval
	c.mu.RUnlock()

	if ok && time.Since(entry.fetchedAt) < refreshInterval {
		return entry.content, nil
	}

	// Memory cache expired or missing - try to refresh
	// This shouldn't happen after prefetch, but handle it gracefully
	if ok {
		// We have stale data, try to refresh in background
		go func() {
			content, err := c.fetchFromGitHub(promptFile)
			if err != nil {
				slog.Warn("failed to refresh instructions",
					"file", promptFile,
					"error", err,
				)
				return
			}
			c.mu.Lock()
			c.cache[promptFile] = &cacheEntry{
				content:   content,
				fetchedAt: time.Now(),
			}
			c.mu.Unlock()
			go func(pf, content string) {
				if err := c.saveToDisk(pf, content); err != nil {
					slog.Warn("failed to save instruction to disk cache",
						"file", pf,
						"error", err,
					)
				}
			}(promptFile, content)
		}()
		// Return stale data for now
		return entry.content, nil
	}

	// No cache at all - this should only happen if prefetch wasn't called
	// Try to load from disk
	content, err := c.loadFromDiskWithExpired(promptFile)
	if err == nil && content != "" {
		c.mu.Lock()
		c.cache[promptFile] = &cacheEntry{
			content:   content,
			fetchedAt: time.Now(),
		}
		c.mu.Unlock()
		return content, nil
	}

	// Last resort: fetch from GitHub
	content, err = c.fetchFromGitHub(promptFile)
	if err != nil {
		return "", err
	}

	c.mu.Lock()
	c.cache[promptFile] = &cacheEntry{
		content:   content,
		fetchedAt: time.Now(),
	}
	c.mu.Unlock()

	go func(pf, content string) {
		if err := c.saveToDisk(pf, content); err != nil {
			slog.Warn("failed to save instruction to disk cache",
				"file", pf,
				"error", err,
			)
		}
	}(promptFile, content)
	return content, nil
}

// loadFromDiskWithExpired loads from disk cache, returning content even if expired.
// Returns content and logs a warning if cache is expired.
func (c *InstructionsCache) loadFromDiskWithExpired(promptFile string) (string, error) {
	cacheDir := CacheDir()
	contentPath := filepath.Join(cacheDir, promptFile)
	metaPath := filepath.Join(cacheDir, promptFile+".meta.json")

	// Check metadata
	metaData, err := os.ReadFile(metaPath)
	if err != nil {
		return "", err
	}

	var meta cacheMeta
	if err := json.Unmarshal(metaData, &meta); err != nil {
		return "", err
	}

	// Read content
	content, err := os.ReadFile(contentPath)
	if err != nil {
		return "", err
	}

	// Check if cache is expired (7 days for disk cache)
	diskCacheTTL := time.Duration(InstructionsDiskCacheTTL) * time.Minute
	if time.Since(meta.FetchedAt) > diskCacheTTL {
		slog.Warn("disk cache expired, using anyway",
			"file", promptFile,
			"age", time.Since(meta.FetchedAt),
		)
	}

	return string(content), nil
}

func (c *InstructionsCache) saveToDisk(promptFile, content string) error {
	if err := EnsureCacheDir(); err != nil {
		return err
	}

	cacheDir := CacheDir()
	contentPath := filepath.Join(cacheDir, promptFile)
	metaPath := filepath.Join(cacheDir, promptFile+".meta.json")

	// Write content
	if err := os.WriteFile(contentPath, []byte(content), 0644); err != nil {
		return err
	}

	// Write metadata
	meta := cacheMeta{
		Version:   c.version,
		FetchedAt: time.Now(),
	}
	metaData, err := json.Marshal(meta)
	if err != nil {
		return err
	}

	return os.WriteFile(metaPath, metaData, 0644)
}

func (c *InstructionsCache) fetchFromGitHub(promptFile string) (string, error) {
	// First, get the latest release tag
	tag, err := c.getLatestReleaseTag()
	if err != nil {
		// Fallback to main branch if release fetch fails
		tag = "main"
	}

	c.version = tag

	// Construct raw GitHub URL
	// Prompts are located at codex-rs/core/{promptFile}
	url := fmt.Sprintf("%s/%s/codex-rs/core/%s",
		GitHubRawBaseURL, tag, promptFile)

	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to fetch instructions: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to fetch instructions: status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read instructions: %w", err)
	}

	return string(body), nil
}

func (c *InstructionsCache) getLatestReleaseTag() (string, error) {
	req, err := http.NewRequest("GET", GitHubReleasesAPI, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Accept", "application/vnd.github+json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to fetch releases: status %d", resp.StatusCode)
	}

	var release struct {
		TagName string `json:"tag_name"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return "", err
	}

	if release.TagName == "" {
		return "", fmt.Errorf("no tag name in release")
	}

	return release.TagName, nil
}
