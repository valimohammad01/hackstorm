# 🔑 Google Gemini API Setup Guide

## What is Gemini API?

Google Gemini is a powerful AI model that can analyze and enhance images. In this surveillance system, it's used to:
- **Enhance blurry or low-quality video frames**
- **Improve weapon detection accuracy**
- **Analyze image clarity and sharpness**
- **Provide AI-powered denoising and sharpening**

## 🚀 Quick Start: Get Your Free API Key

### Step 1: Go to Google AI Studio
Visit: **https://makersuite.google.com/app/apikey**

Or: **https://aistudio.google.com/app/apikey**

### Step 2: Sign In
- Use your Google account (Gmail)
- If you don't have one, create a free Google account

### Step 3: Create API Key
1. Click on **"Create API Key"** button
2. Select **"Create API key in new project"** (recommended)
3. Your API key will be generated instantly

### Step 4: Copy Your API Key
- Click the **Copy** button next to your API key
- It looks like: `AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`
- **Keep it secure!** Don't share it publicly

### Step 5: Use in the System
1. Run the enhanced surveillance system:
   ```bash
   streamlit run main_enhanced.py
   ```
2. In the **sidebar**, find **"Gemini API Key"** input field
3. Paste your API key
4. Click outside the field or press Enter
5. You'll see: ✅ **"Gemini API configured"**

### Step 6: Enable Image Enhancement
1. Check the box: ☑️ **"Enable blur enhancement (Gemini)"**
2. The system will now use AI to enhance blurry frames

## 💰 Pricing Information

### Free Tier
Google Gemini offers a **generous free tier**:
- **15 requests per minute**
- **1 million tokens per day**
- **1,500 requests per day**

For this surveillance system:
- Images are processed every **30 frames** (reduces API calls)
- At 30 FPS, this means **1 API call per second**
- You can run for **hours on the free tier**

### Cost for Paid Usage
If you exceed free tier:
- Gemini 1.5 Flash: **$0.075 per 1 million tokens** (very affordable)
- For image processing: **~$0.001 per image**

## 🔒 Security Best Practices

### DO ✅
- Keep your API key private
- Use environment variables in production
- Regenerate key if accidentally exposed
- Monitor usage in Google Cloud Console

### DON'T ❌
- Share API key publicly
- Commit API key to GitHub/version control
- Use same key across multiple projects (create separate keys)
- Give API key to untrusted people

## 🛠️ Alternative: Environment Variable Setup

For production or repeated use, save API key as environment variable:

### Windows:
```batch
setx GEMINI_API_KEY "your-api-key-here"
```

Then modify the code to use it:
```python
import os
gemini_api_key = os.getenv('GEMINI_API_KEY', '')
```

### Linux/Mac:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Add to `~/.bashrc` or `~/.zshrc` for permanent:
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## 📊 Monitoring API Usage

### Check Usage:
1. Go to: https://aistudio.google.com/
2. Click on your project
3. View usage statistics and quotas

### Rate Limits:
- If you hit rate limits, the system will fallback to basic enhancement
- Error messages appear in console
- Disable enhancement temporarily if needed

## ❓ Common Issues & Solutions

### Issue: "Invalid API Key"
**Solution:**
- Verify you copied the complete key
- Check for extra spaces or characters
- Generate a new key if needed

### Issue: "Quota Exceeded"
**Solution:**
- Wait a few minutes for quota reset
- Reduce enhancement frequency in code
- Upgrade to paid tier if needed

### Issue: "API Key Not Working"
**Solution:**
- Ensure you're signed in to correct Google account
- Check if API is enabled in Google Cloud Console
- Verify internet connection

### Issue: "Slow Processing"
**Solution:**
- API calls add latency (~500ms-2s per frame)
- System processes every 30th frame by default
- Can increase interval or disable enhancement

## 🎯 When to Use Image Enhancement?

### ✅ Good Use Cases:
- Low-quality surveillance cameras
- Nighttime footage
- Blurry or shaky videos
- Long-distance recordings
- Poor lighting conditions

### ❌ Skip Enhancement When:
- High-quality video feed already
- Need maximum FPS/speed
- Working offline
- Limited internet bandwidth
- Processing old footage in bulk

## 🔄 Alternatives to Gemini API

If you can't use Gemini API, the system still works with:

1. **Basic OpenCV Enhancement** (built-in)
   - Sharpening kernel
   - Fast denoising
   - No API required
   - Runs offline

2. **Other Options** (requires code modification):
   - OpenAI Vision API
   - Azure Computer Vision
   - AWS Rekognition
   - Local AI models (Stable Diffusion, etc.)

## 📝 Testing Your Setup

### Quick Test:
1. Run system with API key configured
2. Enable blur enhancement
3. Upload a blurry test video
4. Watch console for Gemini API activity
5. Should see enhanced weapon detection

### Verify It's Working:
- Console shows: "Using Gemini enhancement"
- No API errors in console
- Improved detection on blurry frames
- Slightly slower FPS (normal)

## 🎓 Advanced Configuration

### Adjust Enhancement Frequency
In `main_enhanced.py`, line ~358:
```python
if enable_blur_enhancement and gemini_api_key and frame_count % 30 == 0:
```

Change `30` to:
- `15` for more frequent enhancement (higher API usage)
- `60` for less frequent enhancement (lower API usage)

### Custom Enhancement Prompt
Modify the prompt in `enhance_frame_with_gemini()` function:
```python
prompt = """Your custom instructions for Gemini here"""
```

### Switch Gemini Model
Current: `gemini-1.5-flash` (fast, cheap)
Alternative: `gemini-1.5-pro` (better quality, more expensive)

Change in code:
```python
gemini_model = genai.GenerativeModel('gemini-1.5-pro')
```

## 📚 Additional Resources

- **Gemini API Documentation**: https://ai.google.dev/docs
- **Pricing Details**: https://ai.google.dev/pricing
- **API Reference**: https://ai.google.dev/api
- **Google AI Studio**: https://aistudio.google.com/

## 🎉 You're Ready!

With your Gemini API key set up, you can now:
- ✅ Enhance blurry surveillance footage
- ✅ Improve weapon detection accuracy
- ✅ Process low-quality video feeds
- ✅ Leverage AI for better security

**Run the system and start detecting threats with enhanced accuracy!**

```bash
streamlit run main_enhanced.py
```

---

**Need Help?** Check the main `ENHANCED_SETUP_GUIDE.md` for full system documentation.
