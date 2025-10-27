# Hazard Detection Evaluation Guide

## Metrics Overview

### 1. Binary Detection Metrics (Image-Level)

| Metric | Definition | Target |
|--------|------------|--------|
| **Precision** | TP / (TP + FP) | > 80% |
| **Recall** | TP / (TP + FN) | > 90% (safety-critical) |
| **F1-Score** | Harmonic mean of P & R | > 85% |
| **Accuracy** | (TP + TN) / Total | > 85% |

**Confusion Matrix:**
- **TP (True Positive)**: Correctly detected hazard
- **FP (False Positive)**: Detected hazard when none exists
- **FN (False Negative)**: Missed actual hazard (SAFETY ISSUE!)
- **TN (True Negative)**: Correctly detected no hazard

### 2. Critical Hazard Miss Rate (CHMR) 🚨

**Definition:** Percentage of hazardous scenes completely missed

**Formula:** `CHMR = FN / (TP + FN)`

**Targets:**
- ✅ **< 5%**: Safe for blind navigation (catches ≥95% of hazards)
- ⚠️ **5-10%**: Marginally acceptable (needs improvement)
- ❌ **> 10%**: UNSAFE (too many missed hazards)

**Why it matters:** For a blind navigation system, missing hazards can lead to collisions. CHMR must be minimized.

### 3. Per-Type Metrics

Precision, Recall, and F1 computed for each hazard type:
- `vehicle`
- `trafficcone`
- `creature` (pedestrians)
- `column` (poles, pillars)
- `wall`

**Use case:** Identify which hazard types are problematic (e.g., poor recall on `vehicle` → needs improvement)

### 4. Confidence Calibration

- **Mean Confidence (Correct)**: Average confidence when model is right
- **Mean Confidence (Incorrect)**: Average confidence when model is wrong
- **Confidence Gap**: Difference between the two

**Good calibration:** Model should be MORE confident when correct
- Gap > 10% → ✅ Well calibrated
- Gap < 5% → ⚠️ Poorly calibrated (overconfident on errors)

### 5. Latency Performance

| Metric | Meaning | Target |
|--------|---------|--------|
| **Mean** | Average latency | < 1500ms |
| **Median (P50)** | Typical latency | < 1200ms |
| **P95** | 95% of requests faster than this | < 2000ms |
| **P99** | 99% of requests faster than this | < 2500ms |

**Real-time viability:**
- P95 < 1000ms → ✅ Excellent (sub-second)
- P95 < 2000ms → ✅ Good (acceptable for navigation)
- P95 > 3000ms → ❌ Too slow (unusable)

---

## Running Evaluation

### Quick Start

```bash
python evaluate.py \
  --ground_truth data/ground_truth_labels.json \
  --predictions OUTPUTS \
  --output results/evaluation_report.json
```

### With Different Prediction Versions

```bash
# Evaluate v1.5
python evaluate.py \
  -g data/ground_truth_labels.json \
  -p OUTPUTS/V1 \
  -o results/eval_v1.5.json

# Evaluate v2.0
python evaluate.py \
  -g data/ground_truth_labels.json \
  -p OUTPUTS \
  -o results/eval_v2.0.json
```

---

## Interpreting Results

### Example Output

```
============================================================
BINARY DETECTION PERFORMANCE
------------------------------------------------------------
  Confusion Matrix:
    TP: 18   FP: 2
    FN: 1    TN: 3

  Precision:  90.00%  (How many detections were correct?)
  Recall:     94.74%  (How many hazards did we catch?)
  F1-Score:   92.31%  (Harmonic mean)
  Accuracy:   87.50%  (Overall correctness)

🚨 SAFETY-CRITICAL METRIC
------------------------------------------------------------
  CHMR (Critical Hazard Miss Rate):   5.26%
    → This means 5.3% of hazardous scenes were COMPLETELY MISSED
    ⚠️  CAUTION: 5-10% miss rate (marginally acceptable)
```

**Interpretation:**
- ✅ **Precision 90%**: When model says "hazard", it's right 9/10 times
- ✅ **Recall 94.7%**: Model catches 94.7% of actual hazards
- ⚠️ **CHMR 5.26%**: Borderline acceptable (1 in 19 hazards missed)
- **Action:** Investigate the 1 false negative to reduce CHMR below 5%

### Per-Type Analysis Example

```
PER-HAZARD-TYPE PERFORMANCE
------------------------------------------------------------
Hazard Type     GT Count   Precision    Recall       F1
------------------------------------------------------------
vehicle         45          95.00%      88.89%     91.84%
trafficcone     38          92.31%      92.31%     92.31%
column          32          87.50%      93.75%     90.53%
creature        8           100.00%     75.00%     85.71%
wall            6           66.67%      83.33%     74.07%
```

**Interpretation:**
- `vehicle`: High precision, slightly lower recall → some vehicles missed
- `creature`: Perfect precision, but 25% missed → improve pedestrian detection
- `wall`: Lower precision → some false positives (maybe confused with other objects)

### Latency Analysis Example

```
⏱️  LATENCY PERFORMANCE
------------------------------------------------------------
  Mean:    1245ms  (Average response time)
  Median:  1230ms  (Typical response time)
  P95:     1399ms  (95% faster than this)
  P99:     1489ms  (99% faster than this)
  Range:   1089ms - 1512ms

  Real-time Viability:
    ✅ GOOD: P95 < 2000ms (acceptable for navigation)
```

**Interpretation:**
- ✅ All latencies under 2s → acceptable for navigation
- P95 = 1399ms → 95% of hazard detections complete in < 1.4s
- Consistent performance (range ~400ms) → predictable

---

## Decision Criteria

### Is the model ready for deployment?

**Must-have (blocking issues):**
- ✅ CHMR < 10% (preferably < 5%)
- ✅ Recall > 85%
- ✅ P95 latency < 3000ms

**Should-have (quality issues):**
- ✅ F1-Score > 80%
- ✅ Precision > 80%
- ✅ P95 latency < 2000ms

**Nice-to-have (optimization):**
- F1-Score > 90%
- CHMR < 3%
- P95 latency < 1000ms

### Action Items Based on Results

| Issue | Metric | Action |
|-------|--------|--------|
| High CHMR | CHMR > 5% | Improve recall; analyze false negatives |
| High FP rate | Precision < 80% | Tighten detection threshold; improve prompt |
| Poor type recall | Specific class < 70% | Add more examples of that type to prompt |
| Slow latency | P95 > 2000ms | Use Flash instead of Pro; optimize prompt length |
| Poor calibration | Conf gap < 5% | Adjust temperature; add calibration layer |

---

## Output Files

### 1. Console Output
- Real-time progress during evaluation
- Comprehensive summary with all metrics
- Quick decision summary

### 2. `evaluation_report.json`
```json
{
  "metadata": { ... },
  "detection_performance": {
    "binary_metrics": { ... },
    "confidence_stats": { ... },
    "type_metrics": { ... },
    "error_analysis": { ... }
  },
  "latency_performance": { ... }
}
```

Use for:
- Automated analysis
- Comparison across versions
- Integration into dashboards

---

## Comparing Multiple Runs

```bash
# Run evaluation for both versions
python evaluate.py -g data/ground_truth_labels.json -p OUTPUTS/V1 -o results/eval_v1.5.json
python evaluate.py -g data/ground_truth_labels.json -p OUTPUTS -o results/eval_v2.0.json

# Compare results
python -c "
import json
v1 = json.load(open('results/eval_v1.5.json'))
v2 = json.load(open('results/eval_v2.0.json'))

print('Metric              v1.5      v2.0      Change')
print('='*50)
print(f\"F1-Score:          {v1['detection_performance']['binary_metrics']['f1']:.2%}    {v2['detection_performance']['binary_metrics']['f1']:.2%}    {(v2['detection_performance']['binary_metrics']['f1'] - v1['detection_performance']['binary_metrics']['f1']):.2%}\")
print(f\"CHMR:              {v1['detection_performance']['binary_metrics']['chmr']:.2%}    {v2['detection_performance']['binary_metrics']['chmr']:.2%}    {(v2['detection_performance']['binary_metrics']['chmr'] - v1['detection_performance']['binary_metrics']['chmr']):.2%}\")
"
```

---

## Troubleshooting

### "No common images found"
- Check that image names match between ground truth and predictions
- Ensure prediction files are named: `{image_name}__g15flash__v*.json`

### Type mapping errors
- Edit `TYPE_MAPPING` in `eval/metrics.py` if new types appear
- Check that predicted types align with ground truth types

### Missing latency data
- Ensure predictions were generated with updated `main.py` (includes latency tracking)
- Check for `aggregate_statistics.json` in predictions directory

---

## Next Steps

After running evaluation:
1. **Review CHMR**: If > 5%, analyze false negatives
2. **Check per-type recall**: Focus on types with < 80% recall
3. **Compare prompt versions**: Did v2.0 improve metrics?
4. **Generate visualizations**: Create plots for presentation
5. **Write final report**: Summarize findings and recommendations
