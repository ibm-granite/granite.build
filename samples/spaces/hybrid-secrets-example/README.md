# Hybrid Secret Manager Example

## Description

This example demonstrates how to use the **hybrid secret manager** in gbserver, which chains multiple secret managers together with priority-based fallback. This powerful pattern allows you to:

- Override secrets at runtime using environment variables
- Maintain default secrets in local files for development
- Gradually migrate from one secret backend to another
- Implement defense-in-depth security strategies
- Support both local development and CI/CD with the same configuration

## Prerequisites

1. Create a local secrets directory:
   ```bash
   mkdir -p ~/.gbserver/secrets
   chmod 700 ~/.gbserver/secrets  # Restrict permissions
   ```

2. Add some default secrets to the directory:
   ```bash
   echo "default-api-key" > ~/.gbserver/secrets/api_key
   echo "default-db-password" > ~/.gbserver/secrets/db_password
   chmod 600 ~/.gbserver/secrets/*  # Restrict file permissions
   ```

## Configuration Explanation

### Priority Order

The hybrid manager checks secret sources in order:

1. **Environment variables** (Priority 1 - Highest)
   - Type: `env`
   - Prefix: `GBSERVER_SECRET_`
   - Use case: Runtime overrides, CI/CD injection

2. **Local file-based secrets** (Priority 2 - Fallback)
   - Type: `local`
   - Directory: `~/.gbserver/secrets`
   - Use case: Development defaults, shared team secrets

### How Fallback Works

**`get_secret(name)` - First Match Wins:**
```python
# Checks managers in order, returns first found
secret = manager.get_secret("api_key")
# 1. Checks GBSERVER_SECRET_API_KEY (env)
# 2. If not found, checks ~/.gbserver/secrets/api_key (local)
# 3. Returns first found value or raises error
```

**`get_secrets()` - Merge All:**
```python
# Merges secrets from all managers with priority
secrets = manager.get_secrets()
# 1. Gets all secrets from local (base layer)
# 2. Gets all secrets from env (override layer)
# 3. Merges with priority (env overrides local)
# Returns: {"api_key": "from-env", "db_password": "from-local"}
```

## Usage Instructions

### Setup Local Secrets (Development)

Create default secrets for local development:

```bash
# Create secrets directory
mkdir -p ~/.gbserver/secrets
chmod 700 ~/.gbserver/secrets

# Add common secrets
echo "sk-dev-api-key-12345" > ~/.gbserver/secrets/api_key
echo "dev-db-password" > ~/.gbserver/secrets/db_password
echo "dev-github-token" > ~/.gbserver/secrets/github_token
echo "localhost:5432" > ~/.gbserver/secrets/db_host

# Secure the files
chmod 600 ~/.gbserver/secrets/*
```

### Override with Environment Variables (CI/CD or Testing)

Override specific secrets at runtime without changing files:

```bash
# Override API key for production run
export GBSERVER_SECRET_API_KEY="sk-prod-api-key-67890"

# Override database password for testing
export GBSERVER_SECRET_DB_PASSWORD="test-db-password"

# Run build - will use env overrides + local defaults
gbserver build --space hybrid-secrets-example
```

### Example Scenarios

#### Scenario 1: Pure Local Development

```bash
# No environment variables set
# All secrets come from ~/.gbserver/secrets/

gbserver build --space hybrid-secrets-example
# Uses: api_key from file, db_password from file
```

#### Scenario 2: Override One Secret

```bash
# Override just the API key for testing
export GBSERVER_SECRET_API_KEY="sk-test-api-key"

gbserver build --space hybrid-secrets-example
# Uses: api_key from env (override), db_password from file (default)
```

#### Scenario 3: CI/CD with All Overrides

```bash
# CI/CD pipeline injects all secrets as env vars
export GBSERVER_SECRET_API_KEY="sk-ci-key"
export GBSERVER_SECRET_DB_PASSWORD="ci-password"
export GBSERVER_SECRET_GITHUB_TOKEN="ghp_ci_token"

gbserver build --space hybrid-secrets-example
# Uses: all secrets from env (CI/CD values)
```

#### Scenario 4: Gradual Migration

```bash
# Migrating from local to Vault (future)
# Current: env (overrides) -> local (fallback)
# Future: env (overrides) -> vault (primary) -> local (legacy fallback)

# Add vault manager to hybrid config:
# managers:
#   - type: env
#   - type: vault  # New primary
#   - type: local  # Legacy fallback
```

## Use Cases

### 1. Development with Production-Like Setup

Maintain default secrets in local files, but override for production testing:

```bash
# Local development uses file defaults
gbserver build --space hybrid-secrets-example

# Production testing uses env overrides
export GBSERVER_SECRET_API_KEY="prod-key"
gbserver build --space hybrid-secrets-example
```

### 2. Team Secret Sharing

Share common development secrets via local files (with .gitignore):

```bash
# .gitignore
.gbserver/secrets/

# Team shares setup script
cat > setup-dev-secrets.sh <<EOF
#!/bin/bash
mkdir -p ~/.gbserver/secrets
echo "team-dev-api-key" > ~/.gbserver/secrets/api_key
echo "dev.db.local:5432" > ~/.gbserver/secrets/db_host
chmod 600 ~/.gbserver/secrets/*
EOF
```

### 3. CI/CD with Local Development Support

Same configuration works in both environments:

```yaml
# .github/workflows/build.yml
jobs:
  build:
    steps:
      - name: Run build
        env:
          GBSERVER_SECRET_API_KEY: ${{ secrets.API_KEY }}
          GBSERVER_SECRET_DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
        run: gbserver build --space hybrid-secrets-example
```

```bash
# Local developer setup
mkdir -p ~/.gbserver/secrets
echo "local-dev-key" > ~/.gbserver/secrets/api_key
echo "local-dev-pass" > ~/.gbserver/secrets/db_password

# Same command works locally
gbserver build --space hybrid-secrets-example
```

### 4. Per-Environment Configuration

Different environment variables for different deployments:

```bash
# Development
export GBSERVER_SECRET_API_KEY="dev-key"

# Staging
export GBSERVER_SECRET_API_KEY="staging-key"

# Production
export GBSERVER_SECRET_API_KEY="prod-key"

# Same space config works for all
```

## Testing Priority and Fallback

### Test 1: Verify Local Secrets Work

```bash
# Clear any environment overrides
unset GBSERVER_SECRET_API_KEY
unset GBSERVER_SECRET_DB_PASSWORD

# Create test secret
echo "local-test-value" > ~/.gbserver/secrets/test_secret

# Test retrieval (should use local file)
# In your build, reference {{ secrets.test_secret }}
# Should get: "local-test-value"
```

### Test 2: Verify Environment Override

```bash
# Set both local and env
echo "local-value" > ~/.gbserver/secrets/test_secret
export GBSERVER_SECRET_TEST_SECRET="env-value"

# Test retrieval (should use env override)
# Should get: "env-value" (env wins)
```

### Test 3: Verify Fallback Chain

```bash
# Set only env for one secret
export GBSERVER_SECRET_API_KEY="env-api-key"

# Set only local for another
echo "local-db-pass" > ~/.gbserver/secrets/db_password

# Test get_secrets() merge
# Should get both: {"api_key": "env-api-key", "db_password": "local-db-pass"}
```

### Test 4: Verify Missing Secret Error

```bash
# Clear both sources
unset GBSERVER_SECRET_MISSING
rm -f ~/.gbserver/secrets/missing

# Try to get non-existent secret
# Should raise: SecretNotFoundError
```

## Security Considerations

### Defense-in-Depth

The hybrid approach provides layered security:

1. **Environment layer**: Ephemeral, container-scoped, CI/CD controlled
2. **Local file layer**: Filesystem permissions, developer-controlled
3. **Future layers**: Can add Vault, cloud provider secret managers

### Permission Best Practices

```bash
# Secrets directory: owner read/write/execute only
chmod 700 ~/.gbserver/secrets

# Secret files: owner read/write only
chmod 600 ~/.gbserver/secrets/*

# Verify permissions
ls -la ~/.gbserver/secrets
# Should show: drwx------ for directory
# Should show: -rw------- for files
```

### Gitignore Configuration

**NEVER commit secrets to version control:**

```gitignore
# .gitignore
.gbserver/secrets/
.env
*.secret
*_secret
secrets/
```

### Rotation Strategy

```bash
# 1. Add new secret to env (higher priority)
export GBSERVER_SECRET_API_KEY="new-rotated-key"

# 2. Test with new key
gbserver build --space hybrid-secrets-example

# 3. Update local file after validation
echo "new-rotated-key" > ~/.gbserver/secrets/api_key

# 4. Remove env override
unset GBSERVER_SECRET_API_KEY
```

## Advanced Configuration

### Adding More Secret Managers

Extend the hybrid chain:

```yaml
secret_manager:
  type: hybrid
  config:
    managers:
      # Priority 1: Environment (overrides)
      - type: env
        config:
          prefix: "GBSERVER_SECRET_"

      # Priority 2: HashiCorp Vault (primary production)
      - type: vault
        config:
          url: "https://vault.example.com"
          path: "secret/gbserver"
          token_env: "VAULT_TOKEN"

      # Priority 3: Local files (development fallback)
      - type: local
        config:
          secrets_dir: ~/.gbserver/secrets
```

### Per-Manager Prefix

Use different prefixes for different sources:

```yaml
secret_manager:
  type: hybrid
  config:
    managers:
      # Production overrides
      - type: env
        config:
          prefix: "PROD_SECRET_"

      # Development defaults
      - type: env
        config:
          prefix: "DEV_SECRET_"

      # Local fallback
      - type: local
        config:
          secrets_dir: ~/.gbserver/secrets
```

## Troubleshooting

**Secret not found despite being set:**
- Check manager priority order
- Verify environment variable naming (uppercase, underscores)
- Verify file exists: `ls -la ~/.gbserver/secrets/`
- Check file permissions: `chmod 600 ~/.gbserver/secrets/*`

**Wrong secret value (not getting override):**
- Verify environment variable is exported: `export` (should list it)
- Check spelling/naming exactly matches
- Verify manager order (env should be first for overrides)

**Permission denied errors:**
- Check directory permissions: `chmod 700 ~/.gbserver/secrets`
- Check file permissions: `chmod 600 ~/.gbserver/secrets/*`
- Verify ownership: `ls -la ~/.gbserver/secrets/`

**Secrets not updating:**
- Environment variables: Restart process after changes
- Local files: Changes take effect immediately (re-read on each access)
- Check file modification time: `ls -l ~/.gbserver/secrets/api_key`
