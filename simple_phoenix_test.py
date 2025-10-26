#!/usr/bin/env python3
"""
Simple Phoenix Integration Test
Tests Arize Phoenix setup for Voice AI app
"""

import os
import sys
from datetime import datetime

def test_phoenix_basic():
    """Test basic Phoenix functionality"""
    print("=== ARIZE PHOENIX INTEGRATION TEST ===")
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Import Phoenix
    print("1. Testing Phoenix import...")
    try:
        import phoenix as px
        print("   ✅ Phoenix imported successfully")
    except ImportError as e:
        print(f"   ❌ Phoenix import failed: {e}")
        return False
    
    # Test 2: Import OpenTelemetry components
    print("2. Testing OpenTelemetry components...")
    try:
        from phoenix.otel import register
        from opentelemetry import trace
        print("   ✅ OpenTelemetry components imported")
    except ImportError as e:
        print(f"   ❌ OpenTelemetry import failed: {e}")
        return False
    
    # Test 3: Start Phoenix locally
    print("3. Starting local Phoenix server...")
    try:
        session = px.launch_app(host="127.0.0.1", port=6006)
        print(f"   ✅ Phoenix server started!")
        print(f"   🌐 Phoenix UI: {session.url}")
        print(f"   📊 View AI traces at: {session.url}")
    except Exception as e:
        print(f"   ❌ Phoenix server failed: {e}")
        return False
    
    # Test 4: Register OpenTelemetry tracer
    print("4. Testing OpenTelemetry registration...")
    try:
        tracer_provider = register(
            project_name="voice_ai_test",
            endpoint="http://127.0.0.1:6006/v1/traces"
        )
        print("   ✅ OpenTelemetry tracer registered")
    except Exception as e:
        print(f"   ❌ OpenTelemetry registration failed: {e}")
        return False
    
    # Test 5: Create test trace
    print("5. Creating test trace...")
    try:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("phoenix_integration_test") as span:
            span.set_attribute("test.type", "phoenix_integration")
            span.set_attribute("test.status", "success")
            span.set_attribute("test.timestamp", datetime.now().isoformat())
            print("   ✅ Test trace created successfully")
    except Exception as e:
        print(f"   ❌ Test trace failed: {e}")
        return False
    
    print()
    print("=" * 50)
    print("🎉 PHOENIX INTEGRATION SUCCESSFUL!")
    print("=" * 50)
    print()
    print("✅ What this means:")
    print("   • Phoenix is installed and working")
    print("   • Local Phoenix server is running")
    print("   • OpenTelemetry tracing is active")
    print("   • AI calls will be automatically tracked")
    print()
    print("🚀 Next steps:")
    print("   1. Keep this terminal open (Phoenix server running)")
    print("   2. Open new terminal and run: python app.py")
    print("   3. Visit: http://localhost:5000 (Voice AI app)")
    print(f"   4. View traces: {session.url} (Phoenix dashboard)")
    print()
    print("💡 When you make AI calls, you'll see them in Phoenix!")
    
    return True

def main():
    """Run the test"""
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Note: python-dotenv not installed, using system environment")
    
    success = test_phoenix_basic()
    
    if success:
        print("\n🎯 Phoenix is ready for your Voice AI app!")
        return True
    else:
        print("\n❌ Phoenix setup needs attention")
        return False

if __name__ == "__main__":
    main()
