# TTS Model Evaluation for NavAid

Comprehensive evaluation framework for comparing 5 local Text-to-Speech models for NavAid's audio-first navigation assistant.

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate navaid-tts

# Install system dependencies (macOS)
brew install espeak-ng

# For Linux:
# sudo apt-get install espeak-ng
```

### 2. Run Full Evaluation

```bash
# Complete pipeline: download → preprocess → synthesize → evaluate → visualize
python main.py --mode all

# This will take ~2-3 hours depending on your machine
```

### 3. View Results

Results are saved in `results/`:
- `metrics.json` - All computed metrics
- `comparison.csv` - Tabular comparison
- `summary.md` - Auto-generated recommendations
- `plots/` - Visualization charts
- `mos_samples/` - 30 audio files for human MOS rating

## Evaluated Models

| Model | Type | Expected RTF | Size |
|-------|------|--------------|------|
| Coqui VITS (LJSpeech) | Neural | 0.1-0.15 | ~85MB |
| Coqui VITS (VCTK) | Neural | 0.1-0.15 | ~100MB |
| Coqui Tacotron2 | Neural | 0.2-0.3 | ~120MB |
| Piper | Neural (ONNX) | 0.05-0.08 | ~50MB |
| eSpeak-NG | Formant | 0.01-0.02 | <5MB |

## Evaluation Metrics

### Automated
- **RTF (Real-Time Factor)**: Synthesis time / audio duration. Target: <0.1
- **WER (Word Error Rate)**: Via Whisper ASR round-trip. Target: <10%
- **Footprint**: Disk size, RAM usage, cold start time

### Manual
- **MOS (Mean Opinion Score)**: 1-5 naturalness rating (4 raters, 30 samples)

## Usage Examples

```bash
# Run individual stages
python main.py --mode download       # Download Touchdown/R2R datasets
python main.py --mode preprocess     # Create 100-sample test corpus
python main.py --mode synthesize     # Generate audio for all models
python main.py --mode evaluate       # Compute metrics
python main.py --mode visualize      # Generate plots

# Test specific models only
python main.py --mode all --models coqui_vits_ljspeech piper

# Skip re-generating existing outputs
python main.py --mode synthesize --skip-existing
```

## Directory Structure

```
TTS_METRICS/
├── data/
│   ├── download.py           # Dataset acquisition
│   ├── preprocess.py         # Sentence extraction
│   ├── samples.json          # 100 test sentences
│   └── raw/                  # Downloaded datasets (gitignored)
├── models/
│   ├── base_tts.py           # Abstract TTS interface
│   ├── coqui_*.py            # Coqui implementations
│   ├── piper_tts.py          # Piper implementation
│   ├── espeak_tts.py         # eSpeak wrapper
│   └── cache/                # Model weights (gitignored)
├── eval/
│   ├── metrics.py            # RTF, footprint computation
│   ├── whisper_eval.py       # WER via ASR
│   └── visualize.py          # Plot generation
├── outputs/                  # Generated audio (gitignored)
└── results/
    ├── metrics.json
    ├── comparison.csv
    ├── summary.md
    ├── mos_samples/          # For human rating
    └── plots/
```

## Decision Criteria

**Priority 1: Latency**
- RTF < 0.1 (ideal) or < 0.2 (acceptable)
- Safety-critical warnings must be immediate

**Priority 2: Intelligibility**
- WER < 10% AND MOS > 3.0
- Must be understandable under cognitive load

**Priority 3: Deployment**
- Disk < 500MB, RAM < 1GB
- Must run on mobile devices

## Next Steps

1. Run full evaluation: `python main.py --mode all`
2. Collect MOS ratings from 4 team members using `results/mos_samples/`
3. Update `metrics.json` with MOS scores
4. Review `results/summary.md` for recommendation
5. Integrate selected model into Milestone 2 prototype

## Troubleshooting

**Model download fails:**
- Coqui models download automatically on first run
- Check internet connection and disk space (~500MB needed)

**Out of memory:**
- Run models individually: `--models <model_name>`
- Use smaller test corpus: Edit `data/preprocess.py` to sample fewer sentences

**eSpeak not found:**
- macOS: `brew install espeak-ng`
- Linux: `sudo apt-get install espeak-ng`
- Windows: Download from [eSpeak-NG releases](https://github.com/espeak-ng/espeak-ng/releases)

## References

- Touchdown dataset: [Chen et al., 2020](https://github.com/salesforce/touchdown)
- Room-to-Room: [Anderson et al., 2018](https://github.com/peteanderson80/Matterport3DSimulator)
- Coqui TTS: [GitHub](https://github.com/coqui-ai/TTS)
- Piper: [rhasspy/piper](https://github.com/rhasspy/piper)
