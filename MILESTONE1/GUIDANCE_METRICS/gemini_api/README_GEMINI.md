# Gemini Hazard Detection - Usage Guide

## Rate Limiting for Free Tier

The free tier has **10 RPM (requests per minute)** limit. The code now includes:

1. **Automatic rate limiting** - Spaces requests to stay under limit
2. **429 error handling** - Retries with exponential backoff on rate limit errors
3. **Thread-safe** - Works correctly with concurrent requests

## Quick Start

### 1. Set API Key
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### 2. Test on Single Image
```bash
python main.py \
  --image_path data/Images/191231_14393900006480.png \
  --output_dir outputs/test \
  --rpm_limit 10
```

### 3. Process All Images (Free Tier Optimized)
```bash
python main.py \
  --images_dir data/Images \
  --output_dir outputs/all_images \
  --rpm_limit 10 \
  --max_concurrency 2
```

**Estimated time for 24 images:** ~2.5 minutes (10 RPM limit)

### 4. Process Random Sample
```bash
python main.py \
  --images_dir data/Images \
  --num_samples 5 \
  --output_dir outputs/sample5 \
  --rpm_limit 10
```

## Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--images_dir` | - | Process all images in directory |
| `--image_path` | - | Process single image |
| `--num_samples` | None | Random sample size from images_dir |
| `--output_dir` | **required** | Where to save JSON outputs |
| `--prompt_path` | `prompts/prompt.md` | Custom prompt file |
| `--model` | `gemini-2.5-flash` | Model to use (flash or pro) |
| `--rpm_limit` | **10** | Requests per minute (0 = no limit) |
| `--max_concurrency` | **2** | Parallel requests (2 for free tier) |
| `--temperature` | 0.2 | Sampling temperature |
| `--top_p` | 0.8 | Nucleus sampling |
| `--seed` | 7 | Random seed for sampling |

## Rate Limiting Explained

### How It Works

1. **Per-request delay**: With `rpm_limit=10`, each request waits ~6.6 seconds (60s / 10 * 1.1 safety margin)
2. **Thread-safe**: Multiple threads coordinate to enforce global rate limit
3. **Smart retry**: If 429 error occurs, waits longer (5x normal backoff)

### Free Tier Recommendations

```bash
# Conservative (smoothest, no rate limit errors)
--rpm_limit 10 --max_concurrency 1

# Balanced (recommended, some initial rate limits possible)
--rpm_limit 10 --max_concurrency 2

# Aggressive (may hit rate limits initially, then settle)
--rpm_limit 10 --max_concurrency 4
```

### Paid Tier

If you upgrade to paid tier with higher limits:
```bash
# Example: 360 RPM limit (tier 2)
--rpm_limit 360 --max_concurrency 10

# No rate limiting
--rpm_limit 0 --max_concurrency 20
```

## Output Format

### Per-Image JSON Files

Each image produces a JSON file: `{image_stem}__{model_tag}__v{version}.json`

Example: `191231_14393900006480__g15flash__v2.0.json`

```json
{
  "_meta": {
    "source_image": "/path/to/image.png",
    "model": "gemini-2.5-flash",
    "prompt_version": "2.0",
    "timestamp": "2025-10-27T12:34:56.789Z",
    "latency": {
      "total_ms": 1245.67,
      "api_processing_ms": 1243.12,
      "validation_ms": 2.55
    }
  },
  "result": {
    "hazard_detected": true,
    "num_hazards": 3,
    "hazard_types": ["trafficcone", "vehicle", "person"],
    "one_sentence": "traffic cone, vehicle, and pedestrian blocking walkway.",
    "evasive_suggestion": "multiple hazards ahead—stop and assess before proceeding.",
    "bearing": "center",
    "proximity": "near",
    "confidence": 0.82,
    "notes": "congested scene; limited clearance"
  }
}
```

### Aggregate Statistics File

`aggregate_statistics.json` contains overall performance metrics:

```json
{
  "total_images": 24,
  "successful": 24,
  "failures": 0,
  "total_time_seconds": 145.32,
  "latency_ms": {
    "mean": 1245.67,
    "median": 1230.45,
    "stdev": 89.23,
    "min": 1089.12,
    "max": 1512.34,
    "p50": 1230.45,
    "p95": 1398.67,
    "p99": 1489.23
  },
  "config": {
    "model": "gemini-2.5-flash",
    "rpm_limit": 10,
    "max_concurrency": 2,
    "temperature": 0.2,
    "top_p": 0.8,
    "prompt_version": "2.0"
  },
  "timestamp": "2025-10-27T12:34:56.789Z"
}
```

### Latency Breakdown

- **total_ms**: End-to-end time (API call + validation)
- **api_processing_ms**: Time spent in Gemini API (network + processing)
- **validation_ms**: Local Pydantic validation time (~1-5ms typically)

## Troubleshooting

### Still Getting Rate Limit Errors?

1. **Reduce concurrency**: `--max_concurrency 1`
2. **Lower RPM**: `--rpm_limit 8` (safety margin)
3. **Check dashboard**: Ensure you're not using API elsewhere

### Slow Processing?

With 10 RPM free tier:
- 10 images = ~1 minute
- 24 images = ~2.5 minutes
- 100 images = ~10 minutes

This is expected! Consider upgrading tier if you need faster processing.

### API Errors?

```bash
# Check your API key
echo $GOOGLE_API_KEY

# Verify model access
python -c "import google.generativeai as genai; genai.configure(api_key='$GOOGLE_API_KEY'); print(genai.list_models())"
```

## Latency Metrics

The code tracks three types of latency for each request:

1. **Total Latency** (`total_ms`): Complete end-to-end time
   - Includes API call, network, validation
   - This is what users experience

2. **API Processing** (`api_processing_ms`): Time spent in Gemini
   - Total minus validation time
   - Includes network round-trip + model inference

3. **Validation** (`validation_ms`): Local Pydantic processing
   - Typically 1-5ms (negligible)
   - JSON parsing + schema validation

### Aggregate Statistics Include:

- **Mean/Median**: Average and typical latency
- **Stdev**: Variability in response times
- **P95/P99**: Worst-case latencies (95th/99th percentile)
- **Min/Max**: Best and worst case observed

### Example Summary Output:

```
============================================================
Summary
============================================================
  Processed: 24/24 images
  Failures: 0
  Total time: 145.3s (2.4 min)

  Latency Statistics:
    Mean:   1246ms
    Median: 1230ms
    Stdev:  89ms
    P95:    1399ms
    P99:    1489ms
    Range:  1089ms - 1512ms

  Output dir: outputs/all_v2
  Stats file: outputs/all_v2/aggregate_statistics.json
```

## Next Steps

After generating all predictions:
1. Run evaluation metrics (compare with ground truth)
2. Compute TP, FP, FN, Precision, Recall, F1
3. Calculate CHMR (Critical Hazard Miss Rate)
4. Analyze latency distributions (mean, P95, P99)
5. Generate visualizations

See main project README for evaluation pipeline.
