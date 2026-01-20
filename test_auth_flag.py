"""
Quick test script to verify DISABLE_AUTH feature flag works correctly.
Run this to ensure authentication bypass is functioning.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Set DISABLE_AUTH before importing anything
os.environ["DISABLE_AUTH"] = "true"
os.environ["FERNET_KEY"] = "test_key_" + "x" * 32  # Dummy key for testing

print("Testing DISABLE_AUTH feature flag...")
print("=" * 60)

try:
    from src.config import Config
    config = Config()
    
    print(f"\n✓ Config loaded")
    print(f"  DISABLE_AUTH: {config.DISABLE_AUTH}")
    
    assert config.DISABLE_AUTH == True, "DISABLE_AUTH should be True"
    print(f"  ✓ Feature flag correctly set to True")
    
except Exception as e:
    print(f"\n✗ Config test failed: {e}")
    sys.exit(1)

try:
    from src.auth.auth_service.auth_service import AuthService
    from src.rag_service.dao.factory import get_user_dao
    from src.auth.auth_provider.factory import auth_provider_factory
    
    print(f"\n✓ Auth modules imported")
    
    # Create auth service
    user_dao = get_user_dao()
    auth_service = AuthService(user_dao, auth_provider_factory)
    
    print(f"  ✓ AuthService created")
    
    # Test get_authenticated_user with None (should return default user)
    user = auth_service.get_authenticated_user(None)
    
    print(f"\n✓ Default user retrieved:")
    print(f"  Name: {user.name}")
    print(f"  Email: {user.email}")
    print(f"  Provider: {user.auth_provider}")
    print(f"  ID: {user.id}")
    
    assert user.name == "Default User", "User should be 'Default User'"
    assert user.email == "default@local.dev", "Email should be 'default@local.dev'"
    assert user.auth_provider == "system", "Provider should be 'system'"
    
    print(f"\n✓ Default user validation passed")
    
except Exception as e:
    print(f"\n✗ Auth service test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # Test that auth bypass works
    auth_service.auth(None, "fake_agent_id")
    print(f"\n✓ Auth bypass works (no exception raised)")
    
except Exception as e:
    print(f"\n✗ Auth bypass test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print("\nThe DISABLE_AUTH feature flag is working correctly!")
print("\nTo use in your application:")
print("1. Set DISABLE_AUTH=true in your .env file")
print("2. Restart your server")
print("3. All authentication will be bypassed")
print("4. Default user will be used for all requests")
