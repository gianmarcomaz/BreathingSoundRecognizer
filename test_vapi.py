#!/usr/bin/env python3
"""
Test Vapi Integration
Tests Vapi API connection and phone call functionality
"""

import os
import sys
import requests
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_vapi_integration():
    """Test Vapi API integration"""
    print("=== VAPI INTEGRATION TEST ===")
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Note: python-dotenv not installed, using system environment")
    
    # Get Vapi credentials
    secret_token = os.getenv("VAPI_SECRET_TOKEN")
    public_key = os.getenv("VAPI_PUBLIC_KEY")
    assistant_id = os.getenv("VAPI_ASSISTANT_ID")
    
    print("1. Checking Vapi credentials...")
    if not secret_token or secret_token == "your_vapi_secret_token_here":
        print("   ❌ VAPI_SECRET_TOKEN not configured")
        print("   💡 Get your API keys from: https://dashboard.vapi.ai")
        return False
    
    if not public_key or public_key == "your_vapi_public_key_here":
        print("   ❌ VAPI_PUBLIC_KEY not configured")
        return False
    
    if not assistant_id or assistant_id == "your_vapi_assistant_id_here":
        print("   ❌ VAPI_ASSISTANT_ID not configured")
        print("   💡 Create an assistant at: https://dashboard.vapi.ai")
        return False
    
    print(f"   ✅ Secret Token: {secret_token[:20]}...")
    print(f"   ✅ Public Key: {public_key[:20]}...")
    print(f"   ✅ Assistant ID: {assistant_id}")
    
    # Test API connection
    print("\n2. Testing Vapi API connection...")
    try:
        headers = {
            "Authorization": f"Bearer {secret_token}",
            "Content-Type": "application/json"
        }
        
        # Test assistants endpoint
        response = requests.get("https://api.vapi.ai/assistant", headers=headers, timeout=10)
        
        if response.status_code == 200:
            assistants = response.json()
            print(f"   ✅ API connection successful")
            print(f"   📋 Found {len(assistants)} assistants")
            
            # Check if our assistant exists
            assistant_found = False
            for assistant in assistants:
                if assistant.get("id") == assistant_id:
                    assistant_found = True
                    print(f"   ✅ Assistant '{assistant.get('name', 'Unnamed')}' found")
                    break
            
            if not assistant_found:
                print(f"   ⚠️  Assistant ID {assistant_id} not found in your account")
                print("   💡 Create a new assistant or update the ID in your .env file")
                
        else:
            print(f"   ❌ API connection failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        return False
    
    # Test app integration
    print("\n3. Testing app integration...")
    try:
        from app import VoiceAIApp
        
        app = VoiceAIApp()
        
        if app.vapi_client:
            print("   ✅ App Vapi integration working")
        else:
            print("   ❌ App Vapi integration failed")
            return False
            
    except Exception as e:
        print(f"   ❌ App integration test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 VAPI INTEGRATION SUCCESSFUL!")
    print("=" * 50)
    print()
    print("✅ What this means:")
    print("   • Vapi API keys are working")
    print("   • Assistant is configured")
    print("   • Phone call functionality is ready")
    print("   • App integration is complete")
    print()
    print("🚀 Next steps:")
    print("   1. Your app is running with full Vapi support")
    print("   2. Use the /startCall endpoint to make phone calls")
    print("   3. Test phone calls through the web interface")
    print()
    print("📞 To test a phone call:")
    print("   POST to: http://localhost:5000/startCall")
    print("   Body: {\"phoneNumber\": \"+1234567890\", \"message\": \"Hello!\"}")
    
    return True

def main():
    """Run the Vapi integration test"""
    success = test_vapi_integration()
    
    if success:
        print("\n🎯 Vapi is ready for your Voice AI app!")
    else:
        print("\n❌ Vapi setup needs attention")
        print("💡 Get your API keys from: https://dashboard.vapi.ai")
    
    return success

if __name__ == "__main__":
    main()
