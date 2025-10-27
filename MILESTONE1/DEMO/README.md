# NavAid Demo - Interactive Hazard Detection

Professional Streamlit interface for real-time hazard detection and audio guidance.

## Features

- Upload street-level images
- Real-time hazard detection using Gemini 2.5
- Audio guidance via TTS
- Professional, clean UI
- Model selection (Flash / Flash-Lite / Pro)
- TTS engine selection (System / Coqui)
- Detailed detection results with confidence scores
- JSON output viewer

## Installation

```bash
cd demo
pip install -r requirements.txt
```

### Optional: For Coqui TTS

```bash
pip install TTS scipy
```

## Running the Demo

```bash
export GOOGLE_API_KEY="your_api_key_here"

streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Enter API Key**: Add your Google Gemini API key in the sidebar (or set `GOOGLE_API_KEY` env variable)

2. **Configure Settings**:
   - Select Gemini model (Flash recommended for speed)
   - Choose TTS engine (System or Coqui)
   - Adjust temperature/top_p if needed

3. **Upload Image**:
   - Click "Browse files" and select a street-level image
   - Supported formats: JPG, JPEG, PNG

4. **Analyze**:
   - Click "Analyze Image" button
   - Wait for detection results (typically 1-2 seconds)

5. **Review Results**:
   - View hazard status (detected/clear)
   - Check confidence score and latency
   - Read hazard types and locations
   - Review suggested evasive action

6. **Listen to Guidance**:
   - Click "Play Audio" to hear the guidance
   - TTS reads the evasive suggestion aloud

## UI Components

### Detection Summary
- **Status Card**: Green (clear) or Red (hazard detected)
- **Confidence**: Model's certainty (0-100%)
- **Latency**: Response time in milliseconds

### Hazard Details (if detected)
- **Hazard Types**: List of detected objects (vehicle, cone, etc.)
- **Location**: Bearing (left/center/right)
- **Distance**: Proximity (near/mid/far)
- **Description**: One-sentence summary
- **Suggested Action**: Navigation instructions

### Audio Guidance
- Text display of the guidance message
- Play button for TTS output
- Works with System TTS or Coqui TTS

## Model Options

### Gemini Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| **gemini-2.5-flash** | Fast | Good | Recommended for demo |
| **gemini-2.5-flash-lite** | Fastest | Moderate | Quick testing |
| **gemini-2.5-pro** | Slow | Best | Highest accuracy |

### TTS Engines

| Engine | Quality | Latency | Dependencies |
|--------|---------|---------|--------------|
| **System (pyttsx3)** | Moderate | Fast | Built-in |
| **Coqui TTS** | High | Medium | Requires TTS package |
| **None** | N/A | N/A | Text only, no audio |

## Configuration

### Sidebar Settings

**Basic:**
- Google API Key (required)
- Gemini Model selection
- TTS Engine selection

**Advanced:**
- Temperature (0.0-1.0): Controls randomness
- Top P (0.0-1.0): Controls diversity

## Example Workflow

```
1. User uploads image: street_scene.jpg
2. System analyzes image (1.2 seconds)
3. Results displayed:
   - Hazard Detected
   - 2 hazards: traffic cone, vehicle
   - Location: center
   - Distance: near
   - Confidence: 87%
4. Guidance: "cone and vehicle ahead—stop and navigate carefully around them"
5. User clicks "Play Audio"
6. TTS reads guidance aloud
```

## Troubleshooting

### "Please enter your Google API Key"
- Add API key in sidebar, OR
- Set environment variable: `export GOOGLE_API_KEY="your_key"`

### "TTS failed"
- Check TTS engine is installed: `pip install pyttsx3` or `pip install TTS`
- Try switching to different TTS engine
- Use "None (Text Only)" as fallback

### "Analysis failed"
- Verify API key is correct
- Check internet connection
- Ensure image is valid format (JPG/PNG)
- Try different Gemini model

### Slow performance
- Use `gemini-2.5-flash` instead of Pro
- Check your internet speed
- Reduce image size before upload

## API Rate Limits

**Free Tier:**
- 10 RPM (requests per minute)
- Demo has no rate limiting (manual pacing)

**Paid Tier:**
- Higher limits available
- Adjust `rpm_limit` in code if needed

## Development

### File Structure
```
demo/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Key Functions

- `analyze_image()`: Calls Gemini API for detection
- `TTSEngine`: Unified TTS interface
- `load_prompt()`: Loads prompt template from main project

### Customization

**Change UI Colors:**
Edit CSS in `st.markdown()` section at top of `app.py`

**Add New TTS Engine:**
Extend `TTSEngine` class with new engine type

**Modify Layout:**
Adjust `st.columns()` ratios for different layouts

## Production Deployment

For production use:

1. **Add authentication**: Use Streamlit's auth features
2. **Enable rate limiting**: Set `rpm_limit` in client
3. **Add logging**: Track usage and errors
4. **Optimize images**: Resize/compress before upload
5. **Cache models**: Use `@st.cache_resource` for TTS models
6. **Add analytics**: Track detection accuracy

## License

Part of NavAid project. See root LICENSE file.

## Support

For issues or questions, refer to main project documentation in `MILESTONE1/GUIDANCE_METRICS/`.
