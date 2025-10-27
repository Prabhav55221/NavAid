"""
NavAid: Hazard Detection Demo
A professional interface for real-time hazard detection and audio guidance.
"""

import sys
from pathlib import Path
import streamlit as st
import os
import json
import time
from io import BytesIO
from PIL import Image

# Add GUIDANCE_METRICS directory to path for imports
guidance_metrics_path = Path(__file__).resolve().parent.parent / "GUIDANCE_METRICS"
if str(guidance_metrics_path) not in sys.path:
    sys.path.insert(0, str(guidance_metrics_path))

from gemini_api.gemini_client import GeminiHazardClient
from gemini_api.hazard_schema import HazardOutput, PROMPT_VERSION

# TTS imports
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from TTS.api import TTS
except ImportError:
    TTS = None


# Page configuration
st.set_page_config(
    page_title="NavAid - Hazard Detection",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .hazard-detected {
        border-left-color: #ef4444;
    }
    .no-hazard {
        border-left-color: #10b981;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


class TTSEngine:
    """Unified TTS engine interface."""

    def __init__(self, engine_type: str):
        self.engine_type = engine_type
        self.engine = None
        self._initialize()

    def _initialize(self):
        if self.engine_type == "System (pyttsx3)":
            if pyttsx3:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
        elif self.engine_type == "Coqui TTS":
            if TTS:
                try:
                    self.engine = TTS("tts_models/en/ljspeech/vits")
                except Exception as e:
                    st.error(f"Failed to load Coqui TTS: {e}")

    def speak(self, text: str) -> BytesIO:
        """Generate speech from text."""
        if not self.engine:
            raise RuntimeError(f"TTS engine {self.engine_type} not available")

        if self.engine_type == "System (pyttsx3)":
            # pyttsx3 plays directly, return None
            self.engine.say(text)
            self.engine.runAndWait()
            return None

        elif self.engine_type == "Coqui TTS":
            # Coqui generates audio file
            wav = self.engine.tts(text)

            # Convert to bytes
            import scipy.io.wavfile as wavfile
            import numpy as np

            audio_bytes = BytesIO()
            # Coqui returns numpy array, convert to wav
            wavfile.write(audio_bytes, 22050, np.array(wav))
            audio_bytes.seek(0)
            return audio_bytes


def load_prompt():
    """Load the prompt template."""
    prompt_path = Path(__file__).resolve().parent.parent / "GUIDANCE_METRICS" / "prompts" / "prompt.md"
    try:
        if prompt_path.exists():
            return prompt_path.read_text()
        else:
            print(f"ERROR: Prompt file not found at: {prompt_path}")
            return None
    except Exception as e:
        print(f"ERROR loading prompt: {e}")
        return None


def analyze_image(image: Image.Image, client: GeminiHazardClient, prompt_text: str) -> dict:
    """Analyze image for hazards."""
    # Save image temporarily
    temp_path = Path("/tmp/navaid_temp_image.jpg")
    image.save(temp_path, format='JPEG')

    # Analyze
    start_time = time.time()
    raw_dict, raw_text = client.analyze(temp_path, prompt_text)
    latency_ms = (time.time() - start_time) * 1000

    # Validate and normalize
    result = HazardOutput(**raw_dict).normalized()

    return {
        "result": result.model_dump(),
        "latency_ms": latency_ms,
        "raw_response": raw_text
    }


def main():
    # Header
    st.markdown('<div class="main-header">NavAid: Hazard Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered navigation assistance for the visually impaired</div>', unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.title("Configuration")

    # API Key
    api_key = st.sidebar.text_input(
        "Google API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Enter your Google Gemini API key"
    )

    if not api_key:
        st.warning("Please enter your Google API Key in the sidebar to proceed.")
        st.stop()

    # Model selection
    model_name = st.sidebar.selectbox(
        "Gemini Model",
        options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=0,
        help="Select the Gemini model for hazard detection"
    )

    # TTS Engine selection
    tts_options = []
    if pyttsx3:
        tts_options.append("System (pyttsx3)")
    if TTS:
        tts_options.append("Coqui TTS")
    tts_options.append("None (Text Only)")

    tts_engine_name = st.sidebar.selectbox(
        "TTS Engine",
        options=tts_options,
        index=0 if tts_options else 2,
        help="Select text-to-speech engine for audio output"
    )

    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Prompt Version:** {PROMPT_VERSION}")

    # Load prompt (do this once at startup)
    prompt_text = load_prompt()
    if not prompt_text:
        st.error("Failed to load prompt template. Please check the installation.")
        st.stop()

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader(
            "Upload an image from a pedestrian viewpoint",
            type=["jpg", "jpeg", "png"],
            help="Upload a street-level image for hazard detection"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Detection Results")

        if uploaded_file:
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Initialize client
                        client = GeminiHazardClient(
                            api_key=api_key,
                            model_name=model_name,
                            temperature=temperature,
                            top_p=top_p,
                            rpm_limit=0  # No rate limiting in demo
                        )

                        # Analyze (prompt_text loaded at startup)
                        response = analyze_image(image, client, prompt_text)

                        # Store in session state
                        st.session_state['response'] = response
                        st.rerun()

                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
        else:
            st.info("Upload an image to begin analysis")

    # Display results if available
    if 'response' in st.session_state:
        response = st.session_state['response']
        result = response['result']

        st.markdown("---")

        # Detection summary
        st.subheader("Detection Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            status_class = "hazard-detected" if result['hazard_detected'] else "no-hazard"
            st.markdown(f"""
            <div class="metric-card {status_class}">
                <h3 style="margin:0; color: #{'ef4444' if result['hazard_detected'] else '10b981'};">
                    {'Hazard Detected' if result['hazard_detected'] else 'Path Clear'}
                </h3>
                <p style="margin-top: 0.5rem; color: #6b7280;">
                    {result['num_hazards']} hazard(s) identified
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color: #3b82f6;">Confidence</h3>
                <p style="margin-top: 0.5rem; font-size: 1.5rem; font-weight: 600; color: #1f2937;">
                    {result['confidence']:.0%}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color: #3b82f6;">Latency</h3>
                <p style="margin-top: 0.5rem; font-size: 1.5rem; font-weight: 600; color: #1f2937;">
                    {response['latency_ms']:.0f}ms
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Hazard details
        if result['hazard_detected']:
            st.subheader("Hazard Details")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Hazard Types:**")
                if result['hazard_types']:
                    for hazard_type in result['hazard_types']:
                        st.markdown(f"- {hazard_type.capitalize()}")
                else:
                    st.markdown("- Unknown")

                st.markdown(f"**Location:** {result['bearing'].capitalize()}")
                st.markdown(f"**Distance:** {result['proximity'].capitalize()}")

            with col2:
                st.markdown("**Description:**")
                st.markdown(f"> {result['one_sentence']}")

                st.markdown("**Suggested Action:**")
                st.markdown(f"> {result['evasive_suggestion']}")

        # Audio output
        st.subheader("Audio Guidance")

        audio_text = result['evasive_suggestion']

        col1, col2 = st.columns([3, 1])

        with col1:
            st.text_area("Audio Text", audio_text, height=100, disabled=True)

        with col2:
            if tts_engine_name != "None (Text Only)":
                if st.button("Play Audio", type="secondary"):
                    with st.spinner("Generating audio..."):
                        try:
                            tts_engine = TTSEngine(tts_engine_name)
                            audio_bytes = tts_engine.speak(audio_text)

                            if audio_bytes:
                                st.audio(audio_bytes, format='audio/wav')
                            else:
                                st.success("Audio played through system speaker")

                        except Exception as e:
                            st.error(f"TTS failed: {str(e)}")
            else:
                st.info("TTS disabled")

        # Additional information
        if result.get('notes'):
            st.info(f"**Additional Notes:** {result['notes']}")

        # JSON output (expandable)
        with st.expander("View Raw JSON Response"):
            st.json(result)


if __name__ == "__main__":
    main()
