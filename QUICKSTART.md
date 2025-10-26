# 🚀 Quick Start Guide

Get your Voice AI Application running in 5 minutes!

## 📋 Prerequisites

- Python 3.8+
- API keys for all four services (see setup instructions)

## ⚡ Quick Setup

### 1. Run Setup Script
```bash
python setup.py
```

### 2. Configure API Keys
Edit the `.env` file with your actual credentials:
```bash
# Get these from your service dashboards
LAVA_SECRET_KEY=your_actual_lava_key
LAVA_OPENAI_KEY=your_openai_key_in_lava
PHOENIX_API_KEY=your_phoenix_key
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
VAPI_SECRET_TOKEN=your_vapi_token
VAPI_PUBLIC_KEY=your_vapi_public_key
VAPI_ASSISTANT_ID=your_assistant_id
```

### 3. Activate Environment & Start
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate

# Start the application
python app.py
```

### 4. Test Everything
```bash
python test_integrations.py
```

### 5. Run Demo
```bash
python demo.py
```

## 🌐 Access the App

Open your browser to: **http://localhost:5000**

## 🎯 Quick Test

1. **Text Chat**: Type a message in the web interface
2. **Phone Call**: Enter your phone number and click "Start Call"  
3. **Browser Voice**: Click "Start Voice Chat" and speak

## 🔧 Troubleshooting

**Server won't start?**
- Check your `.env` file has all required keys
- Ensure Python 3.8+ is installed
- Try: `pip install -r requirements.txt`

**API errors?**
- Verify all API keys are correct
- Check service dashboards for account status
- Review the logs for specific error messages

**Webhook issues?**
- Use ngrok for local testing: `ngrok http 5000`
- Update Vapi webhook URL to your ngrok URL + `/vapiWebhook`

## 📞 Getting API Keys

### Lava Payments
1. Sign up at [lavapayments.com](https://lavapayments.com)
2. Get secret key from dashboard
3. Add OpenAI connection and store API key

### Arize Phoenix  
1. Sign up at [arize.com](https://arize.com)
2. Create project and get API key
3. Use endpoint: `https://app.phoenix.arize.com`

### LiveKit
1. Sign up at [livekit.io](https://livekit.io)  
2. Create project and get API key/secret
3. Note your server URL

### Vapi
1. Sign up at [vapi.ai](https://vapi.ai)
2. Create Voice AI Assistant
3. Get public/secret keys and assistant ID
4. Configure webhook URL

## 🎉 You're Ready!

Once everything is running:
- 📞 Make voice calls via phone
- 🌐 Chat via browser voice
- 💰 Track usage in Lava dashboard  
- 📊 Monitor conversations in Phoenix
- 🔄 Scale with LiveKit rooms
- 🤖 Customize AI behavior in Vapi

**Need help?** Check the full README.md or run the demo script!
