#!/usr/bin/env python3
"""
API Key Setup Helper
Use this script to add your Vapi and Arize Phoenix API keys
"""

import os

def update_env_file():
    """Update .env file with new API keys"""
    print("=== API KEY SETUP HELPER ===")
    print()
    
    # Read current .env file
    env_path = ".env"
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    print("Current API key status:")
    
    # Check current values
    vapi_token = None
    vapi_public = None
    vapi_assistant = None
    phoenix_key = None
    
    for line in lines:
        if line.startswith("VAPI_SECRET_TOKEN="):
            vapi_token = line.split("=", 1)[1].strip()
        elif line.startswith("VAPI_PUBLIC_KEY="):
            vapi_public = line.split("=", 1)[1].strip()
        elif line.startswith("VAPI_ASSISTANT_ID="):
            vapi_assistant = line.split("=", 1)[1].strip()
        elif line.startswith("PHOENIX_API_KEY="):
            phoenix_key = line.split("=", 1)[1].strip()
    
    print(f"  Vapi Secret Token: {'✅ Set' if vapi_token and vapi_token != 'your_vapi_secret_token_here' else '❌ Not set'}")
    print(f"  Vapi Public Key: {'✅ Set' if vapi_public and vapi_public != 'your_vapi_public_key_here' else '❌ Not set'}")
    print(f"  Vapi Assistant ID: {'✅ Set' if vapi_assistant and vapi_assistant != 'your_vapi_assistant_id_here' else '❌ Not set'}")
    print(f"  Phoenix API Key: {'✅ Set' if phoenix_key and phoenix_key != 'local_development' else '❌ Using local mode'}")
    print()
    
    # Get new keys from user
    print("Enter your API keys (press Enter to skip):")
    print()
    
    new_vapi_token = input("Vapi Secret Token (sk_...): ").strip()
    new_vapi_public = input("Vapi Public Key (pk_...): ").strip()
    new_vapi_assistant = input("Vapi Assistant ID: ").strip()
    new_phoenix_key = input("Phoenix API Key (phx_...): ").strip()
    
    # Update .env file
    updated = False
    new_lines = []
    
    for line in lines:
        if line.startswith("VAPI_SECRET_TOKEN=") and new_vapi_token:
            new_lines.append(f"VAPI_SECRET_TOKEN={new_vapi_token}\n")
            updated = True
        elif line.startswith("VAPI_PUBLIC_KEY=") and new_vapi_public:
            new_lines.append(f"VAPI_PUBLIC_KEY={new_vapi_public}\n")
            updated = True
        elif line.startswith("VAPI_ASSISTANT_ID=") and new_vapi_assistant:
            new_lines.append(f"VAPI_ASSISTANT_ID={new_vapi_assistant}\n")
            updated = True
        elif line.startswith("PHOENIX_API_KEY=") and new_phoenix_key:
            new_lines.append(f"PHOENIX_API_KEY={new_phoenix_key}\n")
            # Also update endpoint to cloud
            updated = True
        elif line.startswith("PHOENIX_COLLECTOR_ENDPOINT=") and new_phoenix_key:
            new_lines.append("PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com\n")
        else:
            new_lines.append(line)
    
    if updated:
        # Write updated .env file
        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        
        print()
        print("✅ .env file updated successfully!")
        print()
        print("Next steps:")
        print("1. Run: python app.py")
        print("2. Visit: http://localhost:5000")
        if new_phoenix_key:
            print("3. View traces: https://app.phoenix.arize.com")
        else:
            print("3. View traces: http://localhost:6006")
    else:
        print()
        print("No changes made.")

if __name__ == "__main__":
    update_env_file()
