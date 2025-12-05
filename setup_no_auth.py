"""
Quick setup script for disabling authentication in RAGdoll.
This creates/updates your .env file with DISABLE_AUTH=true.
"""

import os
from pathlib import Path

def setup_no_auth():
    """Setup RAGdoll with authentication disabled."""
    
    print("=" * 70)
    print("RAGdoll Authentication Disable Setup")
    print("=" * 70)
    
    env_file = Path(".env")
    
    # Check if .env exists
    if not env_file.exists():
        print("\n⚠ No .env file found. Creating from .env.example...")
        
        example_file = Path(".env.example")
        if example_file.exists():
            content = example_file.read_text()
            env_file.write_text(content)
            print("✓ Created .env from .env.example")
        else:
            print("✗ .env.example not found!")
            print("  Creating minimal .env file...")
            minimal_content = """# RAGdoll Configuration
ENV=dev
DISABLE_AUTH=true
FERNET_KEY=your_fernet_key_here_generate_with_python

# Database (use mock for testing)
MOCK_RAG_DATABASE_SYSTEM=mock

# LLM Configuration
MODEL=gemini
GEMINI_API_KEY=your_gemini_api_key_here
"""
            env_file.write_text(minimal_content)
            print("✓ Created minimal .env file")
    
    # Read current content
    content = env_file.read_text()
    lines = content.split('\n')
    
    # Check if DISABLE_AUTH exists
    has_disable_auth = any('DISABLE_AUTH' in line for line in lines)
    
    if has_disable_auth:
        # Update existing DISABLE_AUTH line
        new_lines = []
        for line in lines:
            if 'DISABLE_AUTH' in line and not line.strip().startswith('#'):
                new_lines.append('DISABLE_AUTH=true')
                print(f"\n✓ Updated existing DISABLE_AUTH to true")
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)
    else:
        # Add DISABLE_AUTH
        # Find a good place to add it (after ENV or at the start)
        insert_index = 0
        for i, line in enumerate(lines):
            if 'ENV' in line:
                insert_index = i + 1
                break
        
        lines.insert(insert_index, '')
        lines.insert(insert_index + 1, '# Disable authentication (for local dev/testing)')
        lines.insert(insert_index + 2, 'DISABLE_AUTH=true')
        
        content = '\n'.join(lines)
        print(f"\n✓ Added DISABLE_AUTH=true to .env")
    
    # Write back
    env_file.write_text(content)
    
    print("\n" + "=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)
    
    print("\nYour .env file now has:")
    print("  DISABLE_AUTH=true")
    print("\nThis means:")
    print("  ✓ No OAuth/Google authentication needed")
    print("  ✓ No JWT tokens required")
    print("  ✓ Default user automatically created")
    print("  ✓ All API endpoints work without auth headers")
    
    print("\nNext steps:")
    print("  1. Start your server: uvicorn src.main:app --reload")
    print("  2. Check status: curl http://localhost:8000/api/auth-status")
    print("  3. Use any endpoint without authentication!")
    
    print("\n⚠ IMPORTANT: Never use DISABLE_AUTH=true in production!")
    print("\nFor more info, see: docs/manuals/authentication_feature_flag.md")
    

if __name__ == "__main__":
    try:
        setup_no_auth()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
