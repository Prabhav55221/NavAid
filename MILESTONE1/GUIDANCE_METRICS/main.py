# main.py
from __future__ import annotations

import argparse, json, os, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from gemini_api.hazard_schema import HazardOutput, OutputEnvelope, PROMPT_VERSION
from gemini_api.gemini_client import GeminiHazardClient

def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()

def list_images(images_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])

def save_json(out_dir: Path, img_path: Path, model_name: str, payload: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    model_tag = "g15flash" if "flash" in model_name else "g15pro"
    fn = f"{stem}__{model_tag}__v{PROMPT_VERSION}.json"
    (out_dir / fn).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir / fn

def process_one(img_path: Path, client: GeminiHazardClient, prompt_text: str, output_dir: Path, model_name: str):
    # call model with detailed timing
    start_time = time.time()
    raw_dict, raw_text = client.analyze(img_path, prompt_text)
    end_time = time.time()

    total_latency_ms = (end_time - start_time) * 1000

    # validation timing
    validation_start = time.time()
    ho = HazardOutput(**raw_dict).normalized()
    validation_ms = (time.time() - validation_start) * 1000

    # Estimate network latency (total - validation)
    # Note: This is approximate; actual API processing time is opaque
    api_processing_ms = total_latency_ms - validation_ms

    env = OutputEnvelope(
        _meta={
            "source_image": str(img_path),
            "model": model_name,
            "prompt_version": PROMPT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency": {
                "total_ms": round(total_latency_ms, 2),
                "api_processing_ms": round(api_processing_ms, 2),
                "validation_ms": round(validation_ms, 2)
            }
        },
        result=ho
    )
    out_path = save_json(output_dir, img_path, model_name, env.model_dump())
    return img_path, out_path, total_latency_ms

def main():
    ap = argparse.ArgumentParser(description="Gemini hazard detector (image -> JSON).")
    ap.add_argument("--images_dir", "-d", type=Path, help="Directory of images to scan.")
    ap.add_argument("--image_path", "-i", type=Path, help="Single image path.")
    ap.add_argument("--num_samples", "-n", type=int, default=None, help="Random sample size from images_dir.")
    ap.add_argument("--output_dir", "-o", type=Path, required=True, help="Where to write JSON files.")
    ap.add_argument("--prompt_path", "-p", type=Path,
                    default=Path("/Users/prabhavsingh/Documents/CLASSES/Fall2025/NAVAID/MILESTONE1/GUIDANCE_METRICS/prompts/prompt.md"))
    ap.add_argument("--model", "-m", default="gemini-2.5-flash",
                    choices=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.0-flash"], help="Gemini model to use.")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--max_concurrency", type=int, default=2,
                    help="Max parallel requests (default: 2 for free tier, use 4+ for paid)")
    ap.add_argument("--rpm_limit", type=int, default=10,
                    help="Requests per minute limit (default: 10 for free tier, 0 = no limit)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # inputs
    prompt_text = read_prompt(args.prompt_path)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    client = GeminiHazardClient(api_key=api_key, model_name=args.model,
                                temperature=args.temperature, top_p=args.top_p,
                                rpm_limit=args.rpm_limit)

    # gather images
    images: List[Path] = []
    if args.image_path:
        if not args.image_path.exists():
            raise FileNotFoundError(args.image_path)
        images = [args.image_path]
    elif args.images_dir:
        paths = list_images(args.images_dir)
        if args.num_samples:
            random.seed(args.seed)
            images = random.sample(paths, min(args.num_samples, len(paths)))
        else:
            images = paths
    else:
        raise SystemExit("Provide --image_path or --images_dir.")

    if not images:
        raise SystemExit("No images found.")

    # concurrency (cap for pro, respect rate limits)
    max_workers = min(args.max_concurrency, 2 if "pro" in args.model else args.max_concurrency)

    # Warn if concurrency might cause rate limit issues
    if args.rpm_limit > 0 and max_workers > args.rpm_limit / 6:
        print(f"⚠️  Warning: max_concurrency={max_workers} with rpm_limit={args.rpm_limit} may cause initial rate limit hits")
        print(f"   Recommended: --max_concurrency {max(1, args.rpm_limit // 10)} for smoother operation\n")

    # Estimate time
    if args.rpm_limit > 0:
        est_minutes = len(images) / args.rpm_limit
        print(f"📊 Processing {len(images)} images at ~{args.rpm_limit} RPM")
        print(f"   Estimated time: {est_minutes:.1f} minutes (~{est_minutes*60:.0f} seconds)\n")

    results = []
    failures = []
    latencies = []

    overall_start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process_one, img, client, prompt_text, args.output_dir, args.model) for img in images]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                img, outp, latency = fut.result()
                results.append((img, outp))
                latencies.append(latency)
                print(f"✓ [{i}/{len(images)}] {img.name} ({latency:.0f}ms)")
            except Exception as e:
                failures.append(str(e))
                print(f"✗ [{i}/{len(images)}] error: {e}")

    total_time = time.time() - overall_start

    # Compute aggregate statistics
    stats = {}
    if latencies:
        import statistics
        sorted_latencies = sorted(latencies)
        stats = {
            "total_images": len(images),
            "successful": len(results),
            "failures": len(failures),
            "total_time_seconds": round(total_time, 2),
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 2),
                "median": round(statistics.median(latencies), 2),
                "stdev": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
                "p50": round(sorted_latencies[len(sorted_latencies)//2], 2),
                "p95": round(sorted_latencies[int(len(sorted_latencies)*0.95)], 2),
                "p99": round(sorted_latencies[int(len(sorted_latencies)*0.99)], 2) if len(sorted_latencies) > 10 else round(max(latencies), 2)
            },
            "config": {
                "model": args.model,
                "rpm_limit": args.rpm_limit,
                "max_concurrency": max_workers,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "prompt_version": PROMPT_VERSION
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Save aggregate stats
        stats_path = args.output_dir / "aggregate_statistics.json"
        stats_path.write_text(json.dumps(stats, indent=2))

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"  Processed: {len(results)}/{len(images)} images")
    print(f"  Failures: {len(failures)}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    if latencies:
        print(f"\n  Latency Statistics:")
        print(f"    Mean:   {stats['latency_ms']['mean']:.0f}ms")
        print(f"    Median: {stats['latency_ms']['median']:.0f}ms")
        print(f"    Stdev:  {stats['latency_ms']['stdev']:.0f}ms")
        print(f"    P95:    {stats['latency_ms']['p95']:.0f}ms")
        print(f"    P99:    {stats['latency_ms']['p99']:.0f}ms")
        print(f"    Range:  {stats['latency_ms']['min']:.0f}ms - {stats['latency_ms']['max']:.0f}ms")
    print(f"\n  Output dir: {args.output_dir}")
    if stats:
        print(f"  Stats file: {args.output_dir}/aggregate_statistics.json")
    if failures:
        print(f"\n  First {min(5, len(failures))} failures:")
        for f in failures[:5]:
            print(f"    - {f}")

if __name__ == "__main__":
    main()
