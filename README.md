# 🎤 Voice AI Assistant - HackAudioFeature

A comprehensive Python Voice AI application integrating **Lava Payments**, **Arize Phoenix**, **LiveKit**, and **Vapi** for a complete voice AI solution with usage billing, observability, real-time communication, and telephony.

## ✅ **READY TO USE - Your API Keys Integrated!**

### 🔥 **Lava Payments + OpenAI Integration Complete**
- **Lava API Key**: `aks_live_ytSy5ImFnc11T4E6RYr3_lA8Oi13ovIGogHxJSN04JkTOatb5BHe0S8` ✅
- **OpenAI API Key**: Integrated for GPT-4o-mini ✅
- **Usage Billing**: Automatic tracking through Lava ✅

## 🚀 **Quick Start (2 Minutes)**

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Copy Environment File**
```bash
copy env_ready.txt .env
```

### 3. **Test Lava Integration**
```bash
python test_lava.py
```

### 4. **Start the Application**
```bash
python app.py
```

### 5. **Open Your Browser**
```
http://localhost:5000
```

## 🎯 **What Works Right Now**

### ✅ **Ready Features**
- **💬 Text Chat**: AI powered by GPT-4o-mini through Lava
- **💰 Usage Billing**: Automatic API cost tracking
- **📊 Health Monitoring**: Service status dashboard
- **🌐 Web Interface**: Beautiful, responsive UI

### ⏳ **Demo Mode (Add API Keys When Ready)**
- **📞 Phone Calls**: Vapi integration ready
- **🎙️ Browser Voice**: LiveKit integration ready  
- **📈 Observability**: Phoenix integration ready

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │────│   Flask Backend  │────│  Lava Payments  │
│ (Text/Voice)    │    │   (AI Logic)     │    │ (Usage Billing) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ OpenAI GPT-4o   │    │   Your Wallet   │
                       │ (AI Responses)  │    │ (Cost Tracking) │
                       └─────────────────┘    └─────────────────┘
```

## 🔧 **API Endpoints**

### **Core Endpoints**
- `GET /` - Main web interface
- `GET /health` - Service status check
- `POST /chat` - AI chat (✅ Working with Lava)

### **Integration Endpoints**
- `GET /getToken` - LiveKit tokens (Demo mode)
- `POST /vapiWebhook` - Vapi webhooks (Demo mode)
- `POST /startCall` - Phone calls (Demo mode)
- `GET /usage` - Lava usage statistics

## 🧪 **Testing**

### **Test Lava Integration**
```bash
python test_lava.py
```
Expected output:
```
✅ Lava API Key is VALID
✅ Lava Forward Endpoint SUCCESS
🎉 Lava integration is working perfectly!
```

### **Test Web Interface**
1. Start app: `python app.py`
2. Open: `http://localhost:5000`
3. Try the text chat feature
4. See real AI responses through Lava!

## 💰 **Cost-Effective Setup**

### **GPT-4o-mini Pricing**
- **Input**: $0.00015 per 1K tokens
- **Output**: $0.0006 per 1K tokens
- **~200x cheaper than GPT-4!**

### **Example Costs**
- **100 chat messages**: ~$0.05
- **1000 chat messages**: ~$0.50
- **Perfect for development and demos**

## 🔑 **Add More API Keys (Optional)**

### **For Full Voice Features**

1. **Arize Phoenix** (AI Observability)
   - Sign up: [arize.com](https://arize.com)
   - Add `PHOENIX_API_KEY` to `.env`

2. **LiveKit** (Real-time Voice)
   - Sign up: [livekit.io](https://livekit.io)
   - Add `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` to `.env`

3. **Vapi** (Phone Calls)
   - Sign up: [vapi.ai](https://vapi.ai)
   - Add `VAPI_SECRET_TOKEN`, `VAPI_PUBLIC_KEY`, `VAPI_ASSISTANT_ID` to `.env`

## 📁 **Project Structure**

```
HackAudioFeature/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── env_ready.txt         # Environment variables (your keys)
├── test_lava.py          # Lava integration test
├── templates/
│   └── index.html        # Web interface
└── README.md             # This file
```

## 🎉 **Demo Features**

### **Working Now**
- **AI Chat**: Ask questions, get responses through Lava
- **Usage Tracking**: See API costs in real-time
- **Service Health**: Monitor integration status

### **Demo Scenarios**
1. **Cost-Effective AI**: "What's the weather like?"
2. **Technical Questions**: "Explain machine learning"
3. **Creative Tasks**: "Write a short poem"
4. **Conversational**: "Tell me a joke"

## 🔍 **Monitoring & Analytics**

### **Lava Dashboard**
- View usage statistics
- Monitor API costs
- Track billing in real-time
- Access at: [app.lavapayments.com](https://app.lavapayments.com)

### **Application Health**
- Service status: `http://localhost:5000/health`
- Integration status for all four services
- Configuration validation

## 🚀 **Deployment Ready**

### **Local Development**
```bash
python app.py  # Development server
```

### **Production Deployment**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

## 🎯 **Next Steps**

1. **✅ Test Current Setup**
   ```bash
   python test_lava.py
   python app.py
   ```

2. **🌐 Try the Web Interface**
   - Open `http://localhost:5000`
   - Test the chat feature
   - See Lava billing in action

3. **📞 Add Voice Features** (Optional)
   - Get Vapi API keys for phone calls
   - Get LiveKit keys for browser voice
   - Get Phoenix keys for observability

4. **🚀 Scale for Production**
   - Deploy to cloud platform
   - Add user authentication
   - Implement usage limits

## 🏆 **Achievement Unlocked**

**✅ Lava Payments Integration Complete**
- Your AI calls are now tracked and billed automatically
- Cost-effective GPT-4o-mini model integrated
- Ready for production scaling
- Zero-setup billing infrastructure

## 📞 **Support**

- **Lava Issues**: [Lava Documentation](https://docs.lavapayments.com)
- **OpenAI Issues**: [OpenAI Platform](https://platform.openai.com)
- **Application Issues**: Check the logs when running `python app.py`

---

**🎉 Your Voice AI Application is ready! Lava + OpenAI integration working perfectly!** 🚀