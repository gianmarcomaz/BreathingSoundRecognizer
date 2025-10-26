# 🔑 Complete API Key Setup Guide

## ✅ **Your Lava API Key (Already Configured)**

Your Lava API key is: `aks_live_ytSy5ImFnc11T4E6RYr3_lA8Oi13ovIGogHxJSN04JkTOatb5BHe0S8`

✅ **This is already integrated in the code!**

## 🚀 **Next Steps - Get Remaining API Keys**

### 1. **🔥 Complete Lava Setup**

**What you need:** OpenAI API key to use with Lava

**Steps:**
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up/login to OpenAI
3. Go to **API Keys** section
4. Create a new API key
5. Copy the key (starts with `sk-...`)
6. This becomes your `LAVA_CONNECTION_SECRET`

### 2. **🔍 Arize Phoenix Setup**

**Steps:**
1. Go to [arize.com](https://arize.com)
2. Sign up for free account
3. Create a new project
4. Go to **Settings** → **API Keys**
5. Generate API key
6. Copy the key → This is your `PHOENIX_API_KEY`
7. Endpoint is: `https://app.phoenix.arize.com`

### 3. **📡 LiveKit Setup**

**Steps:**
1. Go to [livekit.io](https://livekit.io)
2. Sign up for free account
3. Create a new project
4. Go to **Settings** → **Keys**
5. Copy **API Key** → This is your `LIVEKIT_API_KEY`
6. Copy **API Secret** → This is your `LIVEKIT_API_SECRET`
7. Note your server URL → This is your `LIVEKIT_URL`

### 4. **📞 Vapi Setup**

**Steps:**
1. Go to [vapi.ai](https://vapi.ai)
2. Sign up for free account
3. Create a **Voice AI Assistant**
4. Go to **Dashboard** → **API Keys**
5. Copy **Public Key** → This is your `VAPI_PUBLIC_KEY`
6. Copy **Secret Token** → This is your `VAPI_SECRET_TOKEN`
7. Go to **Assistants** → Copy **Assistant ID** → This is your `VAPI_ASSISTANT_ID`

## 📝 **Create Your .env File**

Create a `.env` file in your project root with these values:

```bash
# Lava Payments (✅ Already configured)
LAVA_SECRET_KEY=aks_live_ytSy5ImFnc11T4E6RYr3_lA8Oi13ovIGogHxJSN04JkTOatb5BHe0S8
LAVA_CONNECTION_SECRET=sk-your-openai-api-key-here
LAVA_PRODUCT_SECRET=

# Arize Phoenix
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com
PHOENIX_API_KEY=your_phoenix_api_key_here

# LiveKit
LIVEKIT_API_KEY=your_livekit_api_key_here
LIVEKIT_API_SECRET=your_livekit_secret_here
LIVEKIT_URL=wss://your-project.livekit.cloud

# Vapi
VAPI_SECRET_TOKEN=your_vapi_secret_token_here
VAPI_PUBLIC_KEY=your_vapi_public_key_here
VAPI_ASSISTANT_ID=your_vapi_assistant_id_here

# Flask
FLASK_ENV=development
FLASK_DEBUG=True
```

## 🧪 **Test Your Setup**

### Test Lava Integration First:
```bash
python test_lava.py
```

### Test All Integrations:
```bash
python test_integrations.py
```

### Start the Application:
```bash
python app.py
```

## 🔧 **Troubleshooting**

### **Lava Issues:**
- ✅ **API Key**: Already configured
- ❌ **Connection Secret**: Need your OpenAI API key
- ❌ **Credits**: Make sure you have credits in Lava wallet

### **Common Errors:**

**"Authentication failed"**
- Check your OpenAI API key in `LAVA_CONNECTION_SECRET`

**"Payment required"**
- Add credits to your Lava account at [app.lavapayments.com](https://app.lavapayments.com)

**"Webhook issues"**
- For local testing, use ngrok: `ngrok http 5000`
- Set Vapi webhook URL to: `your-ngrok-url/vapiWebhook`

## 🎯 **Priority Order**

1. **Get OpenAI API key** → Test Lava integration
2. **Get Phoenix API key** → Enable observability  
3. **Get LiveKit credentials** → Enable real-time voice
4. **Get Vapi credentials** → Enable phone calls

## 💡 **Free Tier Information**

- **OpenAI**: $5 free credit for new accounts
- **Lava**: Free tier for testing
- **Phoenix**: Free tier with limited traces
- **LiveKit**: Free tier with usage limits
- **Vapi**: Free trial credits

## 🚀 **Quick Test Commands**

```bash
# 1. Setup project
python setup.py

# 2. Test Lava specifically
python test_lava.py

# 3. Test all services
python test_integrations.py

# 4. Start application
python app.py

# 5. Run demo
python demo.py
```

## 📞 **Need Help?**

1. **Lava Issues**: Check [Lava documentation](https://docs.lavapayments.com)
2. **OpenAI Issues**: Check [OpenAI platform](https://platform.openai.com)
3. **Other Services**: Check respective documentation
4. **General Issues**: Review the logs when running tests

---

**Your Lava integration is ready! Just need the OpenAI key to complete it.** 🔥
