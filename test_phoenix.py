#!/usr/bin/env python3
"""
Test Arize Phoenix Integration
Tests both local and cloud Phoenix setups
"""

import os
import sys
import time
import requests
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_phoenix_local():
    """Test local Phoenix setup"""
    print("🔍 Testing Local Phoenix Integration...")
    print("=" * 50)
    
    try:
        # Test if Phoenix packages are installed
        print("1. Checking Phoenix packages...")
        try:
            import phoenix as px
            from phoenix.otel import register
            print("   ✅ Phoenix packages installed")
        except ImportError as e:
            print(f"   ❌ Phoenix packages missing: {e}")
            print("   💡 Run: pip install arize-phoenix arize-phoenix-otel")
            return False
        
        # Test Phoenix startup
        print("\n2. Starting local Phoenix server...")
        try:
            # Start Phoenix in notebook mode (local)
            session = px.launch_app()
            print(f"   ✅ Phoenix started successfully!")
            print(f"   🌐 Phoenix UI: {session.url}")
            print(f"   📊 View traces at: {session.url}")
            
            # Test if server is responding
            time.sleep(2)  # Give server time to start
            response = requests.get(session.url, timeout=5)
            if response.status_code == 200:
                print("   ✅ Phoenix server responding")
            else:
                print(f"   ⚠️  Phoenix server status: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Phoenix startup failed: {e}")
            return False
        
        # Test OpenTelemetry registration
        print("\n3. Testing OpenTelemetry integration...")
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            
            # Register Phoenix tracer
            tracer_provider = register(
                project_name="voice_ai_test",
                endpoint="http://localhost:6006/v1/traces"
            )
            
            # Create a test trace
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("test_phoenix_integration") as span:
                span.set_attribute("test.type", "phoenix_integration")
                span.set_attribute("test.timestamp", datetime.now().isoformat())
                span.set_attribute("test.status", "success")
                print("   ✅ Test trace created")
            
            print("   ✅ OpenTelemetry integration working")
            
        except Exception as e:
            print(f"   ❌ OpenTelemetry setup failed: {e}")
            return False
        
        print("\n" + "=" * 50)
        print("🎉 LOCAL PHOENIX INTEGRATION SUCCESSFUL!")
        print("=" * 50)
        print(f"📊 Phoenix Dashboard: {session.url}")
        print("💡 You can now see AI traces in real-time!")
        print("🔄 Phoenix will automatically capture LLM calls")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phoenix test failed: {e}")
        return False

def test_phoenix_cloud():
    """Test cloud Phoenix connection (if API key provided)"""
    print("\n🌐 Testing Cloud Phoenix Connection...")
    print("=" * 50)
    
    api_key = os.getenv("PHOENIX_API_KEY")
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    
    if not api_key or api_key == "local_development":
        print("   ⏭️  Skipping cloud test (using local mode)")
        return True
    
    try:
        # Test cloud endpoint
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{endpoint}/health", headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("   ✅ Cloud Phoenix connection successful")
            return True
        else:
            print(f"   ❌ Cloud Phoenix error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Cloud Phoenix test failed: {e}")
        return False

def test_app_integration():
    """Test Phoenix integration in the main app"""
    print("\n🔗 Testing App Integration...")
    print("=" * 50)
    
    try:
        # Import the main app
        from app import VoiceAIApp
        
        print("1. Creating VoiceAIApp instance...")
        app = VoiceAIApp()
        
        print("2. Testing Phoenix setup...")
        app.setup_phoenix()
        
        print("   ✅ App Phoenix integration working")
        return True
        
    except Exception as e:
        print(f"   ❌ App integration failed: {e}")
        return False

def main():
    """Run all Phoenix tests"""
    print("🧪 ARIZE PHOENIX INTEGRATION TEST")
    print("=" * 60)
    print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    results = []
    
    # Test local Phoenix
    results.append(("Local Phoenix", test_phoenix_local()))
    
    # Test cloud Phoenix (if configured)
    results.append(("Cloud Phoenix", test_phoenix_cloud()))
    
    # Test app integration
    results.append(("App Integration", test_app_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Phoenix integration is ready!")
        print("\n💡 Next steps:")
        print("   1. Run: python app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. View traces: http://localhost:6006")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("💡 Try: pip install arize-phoenix arize-phoenix-otel")
    
    return all_passed

if __name__ == "__main__":
    main()
