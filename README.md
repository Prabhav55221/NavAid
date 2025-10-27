# NavAid: AI-Powered Navigation Assistance for the Visually Impaired

NavAid is an intelligent navigation system designed to assist visually impaired users in navigating outdoor and indoor environments safely. The system combines real-time hazard detection using multimodal large language models (MLLMs) with audio-first guidance delivered through optimized text-to-speech (TTS) engines.

## Project Structure

```
NAVAID/
├── MILESTONE1/
│   ├── TTS_METRICS/          # TTS model evaluation framework
│   ├── GUIDANCE_METRICS/     # Hazard detection evaluation
│   └── DEMO/                 # Interactive Streamlit demo
├── NavAid_Proposal.pdf       # Full project proposal
└── README.md                 # This file
```

---

## Components

### 1. TTS Evaluation (`MILESTONE1/TTS_METRICS/`)

Comprehensive evaluation framework for selecting optimal text-to-speech models for navigation audio guidance.

**Evaluated Models:**
- Coqui TTS (VITS - LJSpeech)
- Coqui TTS (VITS - VCTK)
- Coqui TTS (Tacotron2)
- Piper TTS
- eSpeak-NG

**Key Metrics:**
- Real-Time Factor (RTF): Synthesis speed vs. audio duration
- Word Error Rate (WER): Intelligibility via ASR round-trip
- Model Footprint: Disk size, RAM usage, cold start time
- Mean Opinion Score (MOS): Human-rated naturalness

**Directory Structure:**
```
TTS_METRICS/
├── data/                 # Dataset preprocessing
├── models/               # TTS model implementations
├── eval/                 # Evaluation metrics
├── outputs/              # Generated audio files
└── results/              # Metrics, plots, and reports
```

**Quick Start:**
```bash
cd MILESTONE1/TTS_METRICS

# Create conda environment
conda env create -f environment.yml
conda activate tts-metrics

# Download and preprocess datasets
python main.py --mode download
python main.py --mode preprocess

# Synthesize audio for all models
python main.py --mode synthesize

# Evaluate performance
python main.py --mode evaluate

# Generate visualizations
python main.py --mode visualize

# Sample audio for MOS rating
python main.py --mode mos
```

**Full Pipeline:**
```bash
python main.py --mode all
```

**Outputs:**
- `results/metrics.json`: Complete performance metrics
- `results/comparison.csv`: Tabular comparison
- `results/plots/`: RTF, WER, footprint, latency visualizations
- `results/mos_samples/`: Audio samples for human rating

---

### 2. Hazard Detection Evaluation (`MILESTONE1/GUIDANCE_METRICS/`)

End-to-end evaluation system for visual hazard detection using Google Gemini 2.5 Flash multimodal LLM.

**Features:**
- Ground truth label generation from annotated street images
- Real-time hazard detection via Gemini API
- Comprehensive evaluation metrics
- Safety-critical performance analysis
- Latency tracking and optimization

**Directory Structure:**
```
GUIDANCE_METRICS/
├── data/
│   ├── Images/               # Street scene images
│   ├── Annotations/          # JSON annotation files
│   ├── create_labels.py      # Ground truth generator
│   ├── ground_truth_labels.json
│   └── ground_truth_summary.csv
├── gemini_api/
│   ├── gemini_client.py      # API client with rate limiting
│   └── hazard_schema.py      # Output validation schema
├── prompts/
│   └── prompt.md             # Structured detection prompt
├── eval/
│   ├── metrics.py            # Evaluation metrics
│   └── visualize.py          # Result visualization
├── main.py                   # Batch inference script
├── evaluate.py               # Evaluation runner
└── results/                  # Evaluation outputs
```

**Workflow:**

#### Step 1: Generate Ground Truth Labels
```bash
cd MILESTONE1/GUIDANCE_METRICS

python data/create_labels.py
```

**Output:**
- `data/ground_truth_labels.json`: Full annotations
- `data/ground_truth_summary.csv`: Summary table

#### Step 2: Run Hazard Detection
```bash
export GOOGLE_API_KEY="your_api_key_here"

# Process all images
python main.py \
  --images_dir data/Images \
  --output_dir OUTPUTS \
  --model gemini-2.5-flash \
  --rpm_limit 10 \
  --max_concurrency 2
```

**Parameters:**
- `--model`: gemini-2.5-flash (fast) | gemini-2.5-flash-lite (faster) | gemini-2.5-pro (best quality)
- `--rpm_limit`: Requests per minute (10 for free tier, 0 for unlimited)
- `--max_concurrency`: Parallel requests (2 recommended for free tier)
- `--temperature`: Sampling temperature (0.2 default)
- `--top_p`: Nucleus sampling (0.8 default)

**Outputs:**
- Individual JSON files per image with detection results
- `aggregate_statistics.json`: Latency metrics (mean, median, P95, P99)

#### Step 3: Evaluate Performance
```bash
python evaluate.py \
  --ground_truth data/ground_truth_labels.json \
  --predictions OUTPUTS \
  --output results/evaluation_report.json
```

**Evaluation Metrics:**

**Binary Detection:**
- Precision: Correct detections / Total detections
- Recall: Caught hazards / Total actual hazards
- F1-Score: Harmonic mean of precision and recall
- Accuracy: Overall correctness

**Safety-Critical:**
- CHMR (Critical Hazard Miss Rate): Percentage of hazardous scenes completely missed
  - Target: Less than 5%
  - Acceptable: 5-10%
  - Unsafe: Greater than 10%

**Per-Type Analysis:**
- Precision, Recall, F1 for each hazard type (vehicle, trafficcone, creature, column, wall)
- Confusion analysis between predicted and ground truth types

**Confidence Calibration:**
- Mean confidence for correct vs. incorrect predictions
- Calibration gap (should be positive for well-calibrated models)

**Latency:**
- Mean, Median, P95, P99 response times
- Target: P95 less than 2000ms for real-time navigation

#### Step 4: Visualize Results
```bash
# Create comprehensive dashboard
python eval/visualize.py \
  -i results/evaluation_report.json \
  -o results/plots/dashboard.png

# Or generate individual plots
python eval/visualize.py \
  -i results/evaluation_report.json \
  -o results/plots \
  --individual
```

**Generated Visualizations:**
1. Confusion matrix (TP, FP, FN, TN)
2. Metrics comparison (Precision, Recall, F1, Accuracy)
3. CHMR gauge (safety metric with color coding)
4. Per-type performance (grouped bar charts)
5. Confidence distribution (correct vs. incorrect)
6. Latency distribution (min, median, P95, P99, max)

---

### 3. Interactive Demo (`MILESTONE1/DEMO/`)

Professional Streamlit web application for real-time hazard detection with audio guidance.

**Features:**
- Image upload (drag-and-drop or file browser)
- Real-time hazard detection via Gemini API
- Detailed detection results with confidence scores
- Audio guidance via TTS (System or Coqui)
- Model selection (Flash / Flash-Lite / Pro)
- Adjustable inference parameters

**Installation:**
```bash
cd MILESTONE1/DEMO

pip install -r requirements.txt

# Optional: For Coqui TTS
pip install TTS scipy
```

**Running the Demo:**
```bash
export GOOGLE_API_KEY="your_api_key_here"

streamlit run app.py
```

The application will open at `http://localhost:8501`

**Using the Demo:**

1. **Configuration** (Left Sidebar):
   - Enter Google API Key (or set via environment variable)
   - Select Gemini model (gemini-2.5-flash recommended)
   - Choose TTS engine (System or Coqui TTS)
   - Adjust temperature and top_p if needed

2. **Upload Image** (Left Column):
   - Drag and drop or browse for street-level image
   - Supported formats: JPG, JPEG, PNG

3. **Analyze** (Right Column):
   - Click "Analyze Image" button
   - Wait 1-2 seconds for processing

4. **Review Results**:
   - Detection status (Hazard Detected / Path Clear)
   - Confidence percentage
   - Response latency in milliseconds
   - Hazard types, location, and distance
   - Natural language description
   - Suggested evasive action

5. **Audio Guidance**:
   - View guidance text
   - Click "Play Audio" to hear TTS output
   - Audio plays through browser or system speaker

**UI Components:**

**Detection Summary:**
- Status card: Green (clear) or Red (hazard detected)
- Confidence: Model certainty (0-100%)
- Latency: API response time

**Hazard Details (if detected):**
- Types: List of detected objects
- Location: Bearing (left/center/right)
- Distance: Proximity (near/mid/far)
- Description: One-sentence summary
- Suggested Action: Navigation instructions

**Advanced Settings:**
- Temperature: Controls randomness (0.0-1.0)
- Top P: Controls diversity (0.0-1.0)

---

## System Requirements

### Hardware
- CPU: Multi-core processor (Intel i5 or equivalent)
- RAM: 8GB minimum, 16GB recommended
- Disk: 2GB for models and data

### Software
- Python: 3.9 or 3.10
- Operating System: macOS, Linux, or Windows
- Internet: Required for Gemini API calls

### API Requirements
- Google Gemini API key (free tier: 10 RPM, paid tier: higher limits)

---

## Installation

### Option 1: Conda Environment (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd NAVAID

# Create environment for TTS evaluation
cd MILESTONE1/TTS_METRICS
conda env create -f environment.yml
conda activate tts-metrics

# Install guidance metrics dependencies
cd ../GUIDANCE_METRICS
pip install google-generativeai pydantic numpy

# Install demo dependencies
cd ../DEMO
pip install -r requirements.txt
```

### Option 2: Virtual Environment
```bash
cd NAVAID

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r MILESTONE1/TTS_METRICS/requirements.txt
pip install -r MILESTONE1/DEMO/requirements.txt
```

---

## Configuration

### Environment Variables
```bash
# Required for Gemini API
export GOOGLE_API_KEY="your_api_key_here"

# Optional: Adjust rate limits
export GEMINI_RPM_LIMIT=10  # Free tier default
```

### API Key Setup
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Set the environment variable or enter in demo sidebar

---

## Evaluation Metrics Reference

### TTS Metrics

**Real-Time Factor (RTF):**
- Formula: `synthesis_time / audio_duration`
- Target: Less than 0.1 (10x faster than real-time)
- Acceptable: Less than 0.2

**Word Error Rate (WER):**
- Measured via ASR round-trip (TTS → Whisper → compare)
- Target: Less than 10%
- Acceptable: Less than 15%

**Model Footprint:**
- Disk size: Less than 500MB
- RAM usage: Less than 1GB
- Cold start: Less than 3 seconds

### Hazard Detection Metrics

**Precision:** How many detections were correct?
- Formula: `TP / (TP + FP)`
- Target: Greater than 80%

**Recall:** How many hazards did we catch?
- Formula: `TP / (TP + FN)`
- Target: Greater than 90% (safety-critical)

**F1-Score:** Balance of precision and recall
- Formula: `2 * (Precision * Recall) / (Precision + Recall)`
- Target: Greater than 85%

**CHMR (Critical Hazard Miss Rate):**
- Formula: `FN / (TP + FN)`
- Safe: Less than 5%
- Marginal: 5-10%
- Unsafe: Greater than 10%

**Latency:**
- P95: 95% of requests faster than this threshold
- Target: Less than 2000ms for real-time use

---

## Troubleshooting

### TTS Evaluation

**Issue:** OpenMP duplicate library error (macOS)
```bash
# Solution 1: Add to environment.yml
conda install llvm-openmp

# Solution 2: Set environment variable
export KMP_DUPLICATE_LIB_OK=TRUE
```

**Issue:** Piper TTS API errors
```bash
# Skip Piper for now
python main.py --mode all --models coqui_vits_ljspeech coqui_vits_vctk coqui_tacotron espeak
```

### Hazard Detection

**Issue:** Rate limit exceeded (429 error)
```bash
# Reduce concurrency and RPM
python main.py --rpm_limit 8 --max_concurrency 1
```

**Issue:** Slow inference
```bash
# Use Flash-Lite model
python main.py --model gemini-2.5-flash-lite
```

### Demo

**Issue:** NumPy compatibility error with TTS
```bash
# Downgrade NumPy
pip install "numpy<2.0"

# Or use System TTS instead of Coqui
# Select "System (pyttsx3)" in TTS Engine dropdown
```

**Issue:** Prompt template not found
```bash
# Verify path structure
ls MILESTONE1/GUIDANCE_METRICS/prompts/prompt.md

# Check __init__.py exists
ls MILESTONE1/GUIDANCE_METRICS/gemini_api/__init__.py
```

---

## Project Timeline

**Milestone 1: TTS & Guidance Evaluation** (Completed)
- TTS model evaluation framework
- Hazard detection ground truth generation
- Gemini API integration with rate limiting
- Comprehensive evaluation metrics
- Interactive demo application

**Milestone 2: End-to-End Integration** (Future)
- Real-time video stream processing
- Multi-frame temporal consistency
- Path planning with hazard avoidance
- Haptic feedback integration
- Mobile deployment optimization

**Milestone 3: Field Testing** (Future)
- User studies with visually impaired participants
- Real-world navigation scenarios
- Safety validation
- Performance optimization
- Accessibility improvements

---

## Contributing

### Code Structure
- Use type hints for function signatures
- Follow PEP 8 style guidelines
- Add docstrings to all classes and functions
- Write unit tests for new features

### Adding New TTS Models
1. Implement `BaseTTS` interface in `models/`
2. Register model in `model_registry.py`
3. Update evaluation pipeline
4. Document in README

### Adding New Hazard Types
1. Update `CRITICAL_HAZARDS` in `data/create_labels.py`
2. Add to vocabulary in `prompts/prompt.md`
3. Update type mapping in `eval/metrics.py`
4. Re-run evaluation

---

## Citation

If you use NavAid in your research, please cite:

```bibtex
@project{navaid2025,
  title={NavAid: AI-Powered Navigation Assistance for the Visually Impaired},
  author={[Your Name]},
  year={2025},
  institution={[Your Institution]}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## Acknowledgments

- **Datasets:**
  - Touchdown: Outdoor navigation instructions ([Chen et al., 2020](https://github.com/salesforce/touchdown))
  - Room-to-Room (R2R): Indoor navigation instructions ([Anderson et al., 2018](https://github.com/peteanderson80/Matterport3DSimulator))

- **Models:**
  - Google Gemini 2.5 for hazard detection
  - Coqui TTS for high-quality speech synthesis
  - Piper TTS for edge deployment
  - OpenAI Whisper for intelligibility evaluation

- **Tools:**
  - Streamlit for interactive demo
  - Matplotlib/Seaborn for visualizations
  - Pydantic for schema validation

---

## Support

For issues, questions, or contributions:
- Open an issue in the repository
- Refer to component-specific READMEs in subdirectories
- Check troubleshooting section above

---

## Roadmap

**Short-term:**
- Mobile app prototype (iOS/Android)
- Offline mode with local MLLM
- Multi-language TTS support
- Improved prompt engineering

**Long-term:**
- Augmented reality integration
- Social navigation features
- Crowdsourced hazard reporting
- Open-source community platform

---

**Last Updated:** October 2025
