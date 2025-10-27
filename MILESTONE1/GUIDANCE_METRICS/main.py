# main.py
from __future__ import annotations

import argparse, json, os, random
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
    # call model
    raw_dict, raw_text = client.analyze(img_path, prompt_text)

    # validate & normalize
    ho = HazardOutput(**raw_dict).normalized()

    env = OutputEnvelope(
        _meta={
            "source_image": str(img_path),
            "model": model_name,
            "prompt_version": PROMPT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        result=ho
    )
    out_path = save_json(output_dir, img_path, model_name, env.model_dump())
    return img_path, out_path

def main():
    ap = argparse.ArgumentParser(description="Gemini hazard detector (image -> JSON).")
    ap.add_argument("--images_dir", "-d", type=Path, help="Directory of images to scan.")
    ap.add_argument("--image_path", "-i", type=Path, help="Single image path.")
    ap.add_argument("--num_samples", "-n", type=int, default=None, help="Random sample size from images_dir.")
    ap.add_argument("--output_dir", "-o", type=Path, required=True, help="Where to write JSON files.")
    ap.add_argument("--prompt_path", "-p", type=Path,
                    default=Path("/Users/prabhavsingh/Documents/CLASSES/Fall2025/NAVAID/MILESTONE1/GUIDANCE_METRICS/prompts/prompt.md"))
    ap.add_argument("--model", "-m", default="gemini-2.5-flash",
                    choices=["gemini-2.5-flash", "gemini-2.5-pro"], help="Gemini model to use.")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--max_concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # inputs
    prompt_text = read_prompt(args.prompt_path)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    client = GeminiHazardClient(api_key=api_key, model_name=args.model,
                                temperature=args.temperature, top_p=args.top_p)

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

    # concurrency (cap for pro)
    max_workers = min(args.max_concurrency, 2 if "pro" in args.model else args.max_concurrency)

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process_one, img, client, prompt_text, args.output_dir, args.model) for img in images]
        for fut in as_completed(futs):
            try:
                img, outp = fut.result()
                results.append((img, outp))
                print(f"✓ {img.name} -> {outp.name}")
            except Exception as e:
                failures.append(str(e))
                print(f"✗ error: {e}")

    print("\nSummary")
    print(f"  wrote: {len(results)} json files to {args.output_dir}")
    if failures:
        print(f"  failures: {len(failures)}")
        for f in failures[:5]:
            print("   -", f)

if __name__ == "__main__":
    main()
