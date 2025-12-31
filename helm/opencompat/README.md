# OpenCompat Helm Chart

A Helm chart for deploying OpenCompat - a personal API compatibility layer that provides an OpenAI-compatible interface.

## Overview

OpenCompat is a stateful application that acts as a reverse proxy/translation layer between OpenAI-compatible clients and backend services. This Helm chart deploys it to Kubernetes with persistent storage, proper security contexts, and health monitoring.

## Features

- StatefulSet deployment for stable persistent storage
- Separate volumes for critical data (OAuth credentials) and cache (instruction files)
- Security-hardened container (non-root, read-only filesystem, distroless base)
- Comprehensive health probes (startup, liveness, readiness)
- Optional Ingress with TLS support
- Helm tests for deployment validation
- Flexible configuration via values.yaml

## Prerequisites

- Kubernetes 1.23+
- Helm 3.x
- OpenCompat CLI installed locally (for initial authentication)
- PersistentVolume provisioner support (for PVCs)
- (Optional) Ingress controller (nginx recommended)
- (Optional) cert-manager (for TLS certificates)

## Installation

### 1. Authenticate Locally

Before deploying to Kubernetes, you need to authenticate locally:

```bash
opencompat login
```

This will create an `auth.json` file in `~/.local/share/opencompat/`.

### 2. Create Kubernetes Secret

Create a secret from your authentication credentials:

```bash
# Create namespace
kubectl create namespace opencompat

# Create auth secret
kubectl create secret generic opencompat-auth \
  --from-file=auth.json=$HOME/.local/share/opencompat/auth.json \
  -n opencompat
```

### 3. Install the Chart

```bash
helm install opencompat ./helm/opencompat -n opencompat
```

Or with custom values:

```bash
helm install opencompat ./helm/opencompat \
  -n opencompat \
  -f custom-values.yaml
```

### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods -n opencompat

# View logs
kubectl logs -n opencompat -l app.kubernetes.io/name=opencompat -f

# Run Helm tests
helm test opencompat -n opencompat
```

### 5. Access the Service

```bash
# Port-forward to access locally
kubectl port-forward -n opencompat svc/opencompat 8080:8080

# Test health endpoint
curl http://localhost:8080/health

# List available models
curl http://localhost:8080/v1/models
```

## Configuration

### Basic Configuration

The following table lists the main configurable parameters of the chart and their default values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Container image repository | `ghcr.io/edgard/opencompat` |
| `image.tag` | Container image tag | `""` (uses Chart appVersion) |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `config.host` | Server binding address | `0.0.0.0` |
| `config.port` | Server listening port | `8080` |
| `config.verbose` | Enable verbose logging | `false` |
| `config.reasoningEffort` | Reasoning effort level | `medium` |
| `config.reasoningSummary` | Reasoning summary mode | `auto` |
| `config.textVerbosity` | Text verbosity level | `medium` |
| `config.instructionsRefresh` | Cache refresh interval (minutes) | `1440` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `8080` |
| `ingress.enabled` | Enable Ingress | `false` |
| `ingress.className` | Ingress class name | `nginx` |
| `ingress.hosts` | Ingress hosts configuration | `[opencompat.example.com]` |
| `persistence.data.size` | Data volume size | `1Gi` |
| `persistence.cache.size` | Cache volume size | `5Gi` |
| `resources.limits.cpu` | CPU limit | `1000m` |
| `resources.limits.memory` | Memory limit | `512Mi` |
| `resources.requests.cpu` | CPU request | `100m` |
| `resources.requests.memory` | Memory request | `128Mi` |

### Advanced Configuration

#### Custom Storage Class

```yaml
persistence:
  data:
    storageClass: "fast-ssd"
  cache:
    storageClass: "standard"
```

#### Enable Ingress with TLS

```yaml
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: opencompat.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: opencompat-tls
      hosts:
        - opencompat.yourdomain.com
```

#### Adjust Resource Limits

```yaml
resources:
  limits:
    cpu: 2000m
    memory: 1Gi
  requests:
    cpu: 200m
    memory: 256Mi
```

#### Custom OAuth Client ID

```yaml
config:
  oauthClientID: "your-custom-client-id"
```

## Upgrading

To upgrade an existing release:

```bash
helm upgrade opencompat ./helm/opencompat -n opencompat
```

With custom values:

```bash
helm upgrade opencompat ./helm/opencompat \
  -n opencompat \
  -f custom-values.yaml
```

## Uninstalling

To uninstall/delete the deployment:

```bash
helm uninstall opencompat -n opencompat
```

**WARNING:** This will NOT delete the PersistentVolumeClaims. To delete them:

```bash
kubectl delete pvc -n opencompat -l app.kubernetes.io/instance=opencompat
```

## Troubleshooting

### Pod Not Starting

Check logs for errors:

```bash
kubectl logs -n opencompat -l app.kubernetes.io/name=opencompat
```

Common issues:
- Missing `opencompat-auth` secret
- PVC not binding (check storage class)
- Resource limits too low

### Authentication Issues

If you see authentication errors:

1. Verify the secret exists:
   ```bash
   kubectl get secret opencompat-auth -n opencompat
   ```

2. Check if auth.json was copied:
   ```bash
   kubectl exec -n opencompat opencompat-0 -- ls -la /data/opencompat/
   ```

3. Re-create the secret if needed:
   ```bash
   kubectl delete secret opencompat-auth -n opencompat
   kubectl create secret generic opencompat-auth \
     --from-file=auth.json=$HOME/.local/share/opencompat/auth.json \
     -n opencompat
   kubectl rollout restart statefulset/opencompat -n opencompat
   ```

### Slow Startup

The application fetches instruction files from GitHub on startup, which can take 15-30 seconds. The startup probe is configured with a 60-second timeout to accommodate this.

If startup consistently fails:
- Check network connectivity to GitHub
- Increase `startupProbe.failureThreshold` in values.yaml
- Check pod logs for specific errors

### Health Check Failures

Test the health endpoint directly:

```bash
kubectl port-forward -n opencompat svc/opencompat 8080:8080
curl http://localhost:8080/health
```

Expected response: `{"status":"ok"}`

## Architecture

### StatefulSet Design

The chart uses a StatefulSet with:
- Single replica (personal use, one auth.json)
- Ordered pod management
- Persistent volume claim templates for stable storage

### Storage Strategy

Two persistent volumes:

1. **Data volume** (`/data`):
   - Contains OAuth credentials (`auth.json`)
   - Size: 1Gi (actual usage ~10KB)
   - **Critical**: Backup required

2. **Cache volume** (`/cache`):
   - Contains instruction files from GitHub
   - Size: 5Gi
   - **Recoverable**: Regenerates from GitHub if lost

### Security

- Runs as non-root user (UID 65532)
- Read-only root filesystem
- Drops all capabilities
- Distroless container image (minimal attack surface)
- Secrets stored with 0600 permissions

### Health Probes

- **Startup probe**: 60s timeout for GitHub fetch on startup
- **Liveness probe**: Restarts pod if server hangs
- **Readiness probe**: Removes from service during issues

## Examples

### Port-Forward Example

```bash
kubectl port-forward -n opencompat svc/opencompat 8080:8080
```

### API Usage Example

```bash
# List models
curl http://localhost:8080/v1/models

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'

# Streaming completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

### Custom Values Example

Create `custom-values.yaml`:

```yaml
config:
  verbose: "true"
  reasoningEffort: "high"

resources:
  limits:
    cpu: 2000m
    memory: 1Gi

ingress:
  enabled: true
  hosts:
    - host: opencompat.mydomain.com
      paths:
        - path: /
          pathType: Prefix

persistence:
  data:
    storageClass: "fast-ssd"
```

Install with custom values:

```bash
helm install opencompat ./helm/opencompat -n opencompat -f custom-values.yaml
```

## Support

For issues and questions:
- GitHub: https://github.com/edgard/opencompat
- Issues: https://github.com/edgard/opencompat/issues

## License

This chart follows the same license as the OpenCompat project.
