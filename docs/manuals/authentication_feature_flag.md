# Authentication Feature Flag

## Overview

RAGdoll supports a feature flag to disable authentication for local development and testing. When `DISABLE_AUTH=true`, all authentication checks are bypassed and a default user is used for all operations.

## Configuration

### Enable/Disable Authentication

In your `.env` file:

```bash
# Disable authentication (local development/testing)
DISABLE_AUTH=true

# Enable authentication (production - default)
DISABLE_AUTH=false
```

## How It Works

### When `DISABLE_AUTH=true`:

1. **Authentication Bypass**: All auth checks in routes return immediately without validation
2. **Default User**: A system user named "Default User" is automatically created and used
3. **Mock Tokens**: Login/refresh endpoints return mock tokens instead of JWT tokens
4. **Agent Ownership**: All agents are automatically owned by the default user
5. **No OAuth Required**: Google OAuth credentials are not needed

### Default User Details

- **Name**: Default User
- **Email**: default@local.dev
- **Provider**: system
- **Provider ID**: default_user
- **Owned Agents**: Auto-managed (all agents accessible)

## Affected Endpoints

All authentication-related endpoints work differently when auth is disabled:

### `/api/login` (POST)
- **Normal**: Validates OAuth token, creates JWT tokens
- **Disabled**: Returns mock tokens and default user info immediately

### `/api/refresh` (POST)
- **Normal**: Validates refresh token, issues new access token
- **Disabled**: Returns mock token immediately

### `/api/logout` (GET)
- **Normal**: Revokes JWT token
- **Disabled**: Returns success message immediately

### `/api/auth-status` (GET)
- **New endpoint**: Check if authentication is enabled or disabled

### All Protected Routes
- **Normal**: Require valid JWT token in Authorization header
- **Disabled**: No token required, use default user automatically

## Usage Examples

### Starting Server with Auth Disabled

```bash
# Set in .env
DISABLE_AUTH=true

# Start server
uvicorn src.main:app --reload
```

### Testing API Calls

When auth is disabled, you can call endpoints without Authorization headers:

```bash
# Create an agent (no auth needed)
curl -X POST http://localhost:8000/update-agent/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Agent",
    "description": "Testing without auth",
    ...
  }'

# Upload documents (no auth needed)
curl -X POST http://localhost:8000/upload/agent?agent_id=123 \
  -F "file=@document.pdf"
```

### Check Auth Status

```bash
curl http://localhost:8000/api/auth-status
```

Response when disabled:
```json
{
  "auth_enabled": false,
  "auth_disabled": true,
  "message": "Authentication is disabled - using default user"
}
```

## Running the Full Pipeline Script

The `run_full_pipeline.py` script automatically respects the `DISABLE_AUTH` flag:

```bash
# With auth disabled (easier for testing)
DISABLE_AUTH=true python run_full_pipeline.py

# With auth enabled (requires OAuth setup)
DISABLE_AUTH=false python run_full_pipeline.py
```

## Implementation Details

### Config (`src/config.py`)
- Reads `DISABLE_AUTH` environment variable
- Defaults to `false` (auth enabled)

### AuthService (`src/auth/auth_service/auth_service.py`)
- Checks `config.DISABLE_AUTH` in all methods
- Creates/caches default user when needed
- Logs authentication status on startup

### Routes
All route handlers use `auth_service.get_authenticated_user()` which:
- Returns default user when `DISABLE_AUTH=true`
- Performs normal JWT validation when `DISABLE_AUTH=false`

## Security Considerations

⚠️ **WARNING**: Never enable `DISABLE_AUTH` in production environments!

### Safe Use Cases
✅ Local development
✅ Automated testing
✅ CI/CD pipelines
✅ Demos and proof-of-concepts

### Unsafe Use Cases
❌ Production deployments
❌ Public-facing APIs
❌ Shared development environments
❌ Any system with real user data

## Logging

When auth is disabled, you'll see log messages:

```
🔓 AUTHENTICATION DISABLED - Using default user for all requests
🔓 Login bypassed - authentication disabled
🔓 Auth check bypassed for agent xyz123
🔓 Returning default user (auth disabled)
```

When auth is enabled:

```
🔒 Authentication enabled
```

## Troubleshooting

### Issue: "Default user not found"
**Solution**: The user is created automatically on first use. Check database permissions.

### Issue: "Agent ownership errors"
**Solution**: Ensure `DISABLE_AUTH=true` is set in your environment, not just .env file.

### Issue: "Still getting 401 errors"
**Solution**: 
1. Verify `DISABLE_AUTH=true` in your .env
2. Restart the server
3. Check logs for "AUTHENTICATION DISABLED" message
4. Try the `/api/auth-status` endpoint

## Migration Guide

### From Auth Enabled → Disabled

1. Set `DISABLE_AUTH=true` in `.env`
2. Restart server
3. Default user is created automatically
4. All existing agents remain accessible

### From Auth Disabled → Enabled

1. Set `DISABLE_AUTH=false` in `.env`
2. Configure OAuth credentials (Google)
3. Set `JWT_TOKEN_SECRET`
4. Restart server
5. Users must log in via OAuth
6. Reassign agent ownership as needed

## Related Files

- `src/config.py` - Feature flag configuration
- `src/auth/auth_service/auth_service.py` - Authentication bypass logic
- `src/routes/auth.py` - Auth endpoint modifications
- `src/globals.py` - AuthService initialization
- `.env.example` - Configuration template

## Future Enhancements

Potential improvements to the auth system:

- [ ] Support multiple default users
- [ ] Custom default user configuration
- [ ] Auth simulation mode (test with fake tokens)
- [ ] Per-endpoint auth override
- [ ] Detailed auth metrics/logging
