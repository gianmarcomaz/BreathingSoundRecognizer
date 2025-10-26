#!/usr/bin/env python3
"""
Lava Payments Integration Test
Test the Lava API integration with your specific API keys
"""

import os
import json
import base64
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LavaAPITester:
    def __init__(self):
        # Load from environment instead of hardcoding
        self.lava_secret_key = os.getenv("LAVA_SECRET_KEY")
        self.lava_connection_secret = os.getenv("LAVA_CONNECTION_SECRET")
        self.base_url = "https://api.lavapayments.com/v1"

        # Safety check: warn if keys missing
        if not self.lava_secret_key or not self.lava_connection_secret:
            print("❌ Missing keys: set LAVA_SECRET_KEY and LAVA_CONNECTION_SECRET in .env")
        
    def test_api_key_validation(self):
        """Test if the Lava API key is valid"""
        print("🔑 Testing Lava API Key Validation...")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.lava_secret_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.base_url}/usage",
                headers=headers,
                timeout=10
            )
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Lava API Key is VALID")
                return True
            elif response.status_code == 401:
                print("❌ Lava API Key is INVALID")
                return False
            else:
                print(f"✅ Lava API Key appears valid (status: {response.status_code})")
                return True
                
        except Exception as e:
            print(f"❌ API Key test failed: {e}")
            return False
    
    def test_forward_endpoint(self):
        """Test the Lava forward endpoint for AI proxy"""
        print("\n🚀 Testing Lava Forward Endpoint...")
        
        try:
            # Create compound auth token for forward endpoint
            auth_payload = {
                "secret_key": self.lava_secret_key,
                "connection_secret": self.lava_connection_secret
            }
            
            token = base64.b64encode(json.dumps(auth_payload).encode()).decode()
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "x-lava-metadata": json.dumps({
                    "test": "lava_integration",
                    "source": "test_script"
                })
            }
            
            # Simple test request using GPT-4o-mini (cheaper)
            request_data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Respond briefly."
                    },
                    {
                        "role": "user",
                        "content": "Say 'Hello from Lava!' in exactly those words."
                    }
                ],
                "max_tokens": 20,
                "temperature": 0.1
            }
            
            print(f"   Sending request to: {self.base_url}/forward")
            print(f"   Model: {request_data['model']}")
            print(f"   Message: {request_data['messages'][1]['content']}")
            
            response = requests.post(
                f"{self.base_url}/forward",
                json=request_data,
                headers=headers,
                timeout=30
            )
            
            print(f"   Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    ai_response = result['choices'][0].get('message', {}).get('content', '')
                    usage = result.get('usage', {})
                    
                    print("✅ Lava Forward Endpoint SUCCESS")
                    print(f"   AI Response: {ai_response}")
                    print(f"   Usage: {usage}")
                    print("   🔥 Your Lava + OpenAI integration is working!")
                    
                    return True
                else:
                    print("❌ Invalid response format from Lava")
                    print(f"   Response: {response.text}")
                    return False
                    
            elif response.status_code == 401:
                print("❌ Authentication failed")
                print("💡 Check your OpenAI API key in LAVA_CONNECTION_SECRET")
                return False
            elif response.status_code == 402:
                print("❌ Payment required - Insufficient wallet balance")
                print("💡 Add credits to your Lava account")
                return False
            else:
                print(f"❌ Forward endpoint failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Forward endpoint test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all Lava integration tests"""
        print("🔥 LAVA PAYMENTS INTEGRATION TEST")
        print("=" * 50)

        scrubbed_secret = (self.lava_secret_key[:5] + "...") if self.lava_secret_key else "MISSING"
        scrubbed_conn   = (self.lava_connection_secret[:5] + "...") if self.lava_connection_secret else "MISSING"

        print(f"Lava Key: {scrubbed_secret}")
        print(f"Conn Key: {scrubbed_conn}")
        print(f"Base URL: {self.base_url}")
        
        results = {}
        ...
        # Summary
        print("\n" + "=" * 50)
        print("📋 LAVA TEST SUMMARY")
        print("=" * 50)
        
        for test, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test.upper():<12} {status}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if results.get('forward'):
            print("🎉 Lava integration is working perfectly!")
            print("💰 Your AI calls will be tracked and billed through Lava")
            print("🤖 Ready to use GPT-4o-mini through Lava proxy")
            return True
        else:
            print("🔧 Lava integration needs attention")
            
            if not results.get('api_key'):
                print("   • Check your LAVA_SECRET_KEY")
            if not results.get('forward'):
                print("   • Check your OpenAI API key")
                print("   • Ensure you have credits in your Lava wallet")
            
            return False

def main():
    """Main test function"""
    print("🧪 Lava Payments Integration Test")
    print("Testing your integrated API keys")
    print()
    
    tester = LavaAPITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Ready to use Lava for AI billing!")
        print("💡 Next steps:")
        print("   1. Run: python app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Try the chat feature!")
    else:
        print("\n🔧 Please fix the issues above and try again")

if __name__ == "__main__":
    main()