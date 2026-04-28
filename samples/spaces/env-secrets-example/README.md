# Environment Variable Secret Manager Example

## Description

This example demonstrates how to use the **environment variable secret manager** (`env` type) in gbserver. The env secret manager retrieves secrets from environment variables, making it ideal for cloud-native deployments, CI/CD pipelines, and containerized environments.

## Prerequisites

None - this secret manager only requires that you set the appropriate environment variables before running gbserver commands.

## Usage Instructions

### 1. Set Environment Variables

Set secrets as environment variables with the `GBSERVER_SECRET_` prefix:

```bash
# API keys
export GBSERVER_SECRET_API_KEY="your-api-key-here"
export GBSERVER_SECRET_GITHUB_TOKEN="ghp_xxxxxxxxxxxx"

# Database credentials
export GBSERVER_SECRET_DB_PASSWORD="secure-password"
export GBSERVER_SECRET_DB_USERNAME="admin"

# Cloud service credentials
export GBSERVER_SECRET_AWS_ACCESS_KEY="AKIAIOSFODNN7EXAMPLE"
export GBSERVER_SECRET_AWS_SECRET_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
```

### 2. Use the Space Configuration

This space is configured to use the env secret manager. Simply reference secrets in your build configurations:

```yaml
# In your build.yaml or step.yaml
params:
  api_key: "{{ secrets.api_key }}"
  github_token: "{{ secrets.github_token }}"
  db_password: "{{ secrets.db_password }}"
```

### 3. Test the Configuration

Verify that secrets are being loaded correctly:

```bash
# Set a test secret
export GBSERVER_SECRET_TEST_KEY="test-value"

# Run a build that uses the secret
gbserver build --space env-secrets-example
```

## Name Transformation Examples

The env secret manager automatically transforms secret names to environment variable names:

| Secret Name (in config) | Environment Variable Name | Example Value |
|-------------------------|---------------------------|---------------|
| `api_key` | `GBSERVER_SECRET_API_KEY` | `sk-abc123...` |
| `api-key` | `GBSERVER_SECRET_API_KEY` | `sk-abc123...` |
| `api.key` | `GBSERVER_SECRET_API_KEY` | `sk-abc123...` |
| `db_password` | `GBSERVER_SECRET_DB_PASSWORD` | `secure-pass` |
| `github/token` | `GBSERVER_SECRET_GITHUB_TOKEN` | `ghp_xxx...` |
| `aws.access.key` | `GBSERVER_SECRET_AWS_ACCESS_KEY` | `AKIA...` |

**Note**: All special characters (`.`, `-`, `/`, etc.) are converted to underscores (`_`), and the name is uppercased.

## Use Cases

The environment variable secret manager is ideal for:

1. **CI/CD Pipelines**: GitHub Actions, GitLab CI, Jenkins, etc. can inject secrets as environment variables
2. **Containerized Deployments**: Docker and Kubernetes can pass secrets as environment variables
3. **Cloud Platforms**: Most cloud platforms (AWS ECS, Google Cloud Run, Azure Container Apps) support environment variable injection
4. **Local Development**: Simple setup for developers who don't need complex secret management
5. **Testing**: Easy to set up temporary secrets for testing without touching files or external services

## Security Notes

**Important Security Considerations:**

- Environment variables are **visible to the process and its children**
- Environment variables may be **logged in process listings** (e.g., `ps aux`)
- Environment variables are **stored in memory** and may be swapped to disk
- Suitable for containerized environments where process isolation is enforced
- **Do NOT use** in shared multi-user systems where processes are visible to other users
- Consider using more secure secret managers (vault, cloud provider secret managers) for production systems handling highly sensitive data

**Best Practices:**

- Use environment variables for **ephemeral workloads** (containers, CI/CD jobs)
- Ensure proper **RBAC/IAM policies** control who can inject environment variables
- **Rotate secrets regularly** and update environment configurations
- Use **minimal secrets** - only expose what's needed for each workload
- Consider **hybrid approach** (see hybrid-secrets-example) for defense-in-depth

## Example: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
CMD ["gbserver", "build", "--space", "env-secrets-example"]
```

```bash
# Run with secrets
docker run \
  -e GBSERVER_SECRET_API_KEY="sk-abc123" \
  -e GBSERVER_SECRET_DB_PASSWORD="secure-pass" \
  your-image:latest
```

## Example: Kubernetes Deployment

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gbserver-build
spec:
  containers:
  - name: gbserver
    image: your-image:latest
    env:
    - name: GBSERVER_SECRET_API_KEY
      valueFrom:
        secretKeyRef:
          name: gbserver-secrets
          key: api-key
    - name: GBSERVER_SECRET_DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: gbserver-secrets
          key: db-password
```

## Troubleshooting

**Secret not found:**
- Verify the environment variable is set: `echo $GBSERVER_SECRET_API_KEY`
- Check the name transformation is correct (uppercase, underscores)
- Ensure the prefix matches (`GBSERVER_SECRET_` by default)

**Permission errors:**
- Ensure the process has permission to read environment variables
- In containerized environments, verify secret injection configuration

**Secrets not updating:**
- Environment variables are read at process start
- Restart the process after changing environment variables
