#!/usr/bin/env python3
"""
Quick Vapi Key Update Script
Based on Berkeley AI Hackathon winner's implementation
"""

import os

def update_vapi_keys():
    """Update .env file with Vapi keys"""
    print("=== VAPI KEY UPDATE (Berkeley Winner Pattern) ===")
    print()
    
    # Get keys from user
    print("Enter your Vapi API keys from dashboard.vapi.ai:")
    print()
    
    secret_token = input("Private API Key (sk_...): ").strip()
    public_key = input("Public API Key (pk_...): ").strip()
    assistant_id = input("Assistant ID: ").strip()
    
    if not secret_token or not public_key or not assistant_id:
        print("❌ All keys are required!")
        return False
    
    # Read current .env
    with open('.env', 'r') as f:
        content = f.read()
    
    # Update Vapi keys
    content = content.replace('VAPI_SECRET_TOKEN=your_vapi_secret_token_here', f'VAPI_SECRET_TOKEN={secret_token}')
    content = content.replace('VAPI_PUBLIC_KEY=your_vapi_public_key_here', f'VAPI_PUBLIC_KEY={public_key}')
    content = content.replace('VAPI_ASSISTANT_ID=your_vapi_assistant_id_here', f'VAPI_ASSISTANT_ID={assistant_id}')
    
    # Write updated .env
    with open('.env', 'w') as f:
        f.write(content)
    
    print()
    print("✅ Vapi keys updated successfully!")
    print()
    print("🚀 Next steps:")
    print("1. Your app will automatically detect the new keys")
    print("2. Restart your app: python app.py")
    print("3. Test phone calls at: http://localhost:5000")
    print()
    print("📞 Your Voice AI app now has full Vapi integration!")
    print("   (Following the Berkeley AI Hackathon winner's pattern)")
    
    return True

if __name__ == "__main__":
    update_vapi_keys()
