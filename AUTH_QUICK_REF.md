# 🔓 Authentication Feature Flag - Quick Reference

## TL;DR

Set `DISABLE_AUTH=true` in your `.env` file to disable all authentication in RAGdoll.

## Quick Setup

```bash
# Option 1: Run the setup script
python setup_no_auth.py

# Option 2: Manual setup
echo "DISABLE_AUTH=true" >> .env
```

## What It Does

| Feature | Auth Enabled | Auth Disabled |
|---------|-------------|---------------|
| Login Required | ✅ Yes (OAuth) | ❌ No |
| JWT Tokens | ✅ Required | ❌ Not needed |
| User Management | ✅ Real users | ⚡ Default user |
| Agent Ownership | ✅ Per user | ⚡ All accessible |
| API Headers | ✅ Authorization required | ❌ None needed |

## Environment Variable

```bash
# .env file
DISABLE_AUTH=true   # ← Add this line
```

## Default User (Auto-created)

```json
{
  "name": "Default User",
  "email": "default@local.dev",
  "auth_provider": "system",
  "owned_agents": []  // Auto-managed
}
```

## Example Usage

### With Auth Disabled

```bash
# No headers needed!
curl -X POST http://localhost:8000/update-agent/ \
  -H "Content-Type: application/json" \
  -d '{"name": "My Agent", ...}'
```

### With Auth Enabled

```bash
# Authorization header required
curl -X POST http://localhost:8000/update-agent/ \
  -H "Authorization: Bearer <jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Agent", ...}'
```

## Check Status

```bash
curl http://localhost:8000/api/auth-status
```

Response:
```json
{
  "auth_enabled": false,
  "auth_disabled": true,
  "message": "Authentication is disabled - using default user"
}
```

## Affected Endpoints

### Auth Endpoints
- ✅ `/api/login` - Returns mock tokens
- ✅ `/api/refresh` - Returns mock tokens  
- ✅ `/api/logout` - Returns success
- ✅ `/api/auth-status` - Shows current status

### Protected Endpoints (No auth needed when disabled)
- ✅ `/update-agent/` - Create/update agents
- ✅ `/agents/` - List agents
- ✅ `/delete-agent` - Delete agents
- ✅ `/upload/agent` - Upload documents
- ✅ `/documents/agent` - List documents
- ✅ `/delete-document` - Delete documents
- ✅ `/api-keys/*` - Manage API keys

## Use Cases

### ✅ Good For
- Local development
- Automated testing
- CI/CD pipelines
- Demos
- Prototyping

### ❌ Bad For
- Production deployments
- Public APIs
- Real user data
- Shared environments

## Troubleshooting

### Still getting 401 errors?

1. Check .env has `DISABLE_AUTH=true`
2. Restart server
3. Verify with `/api/auth-status`
4. Check logs for "🔓 AUTHENTICATION DISABLED"

### Default user not working?

Database permissions issue - check MongoDB/mock DB access

## Files to Know

| File | Purpose |
|------|---------|
| `.env` | Config (add `DISABLE_AUTH=true`) |
| `setup_no_auth.py` | Auto-setup script |
| `docs/manuals/authentication_feature_flag.md` | Full docs |
| `AUTHENTICATION_IMPLEMENTATION.md` | Implementation summary |

## Commands

```bash
# Setup
python setup_no_auth.py

# Start server
uvicorn src.main:app --reload

# Check status
curl http://localhost:8000/api/auth-status

# Test endpoint (no auth)
curl -X GET http://localhost:8000/agents/
```

## Security Warning

⚠️ **NEVER** set `DISABLE_AUTH=true` in production!

## More Info

- Full documentation: `docs/manuals/authentication_feature_flag.md`
- Implementation details: `AUTHENTICATION_IMPLEMENTATION.md`
- Code: `src/auth/auth_service/auth_service.py`
