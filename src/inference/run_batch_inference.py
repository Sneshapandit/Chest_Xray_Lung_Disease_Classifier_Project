import argparse
import csv
from datetime import datetime
from pathlib import Path
import runpy

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classifier(script_path: Path):
    mod = runpy.run_path(str(script_path))
    if "classify_image" not in mod:
        raise RuntimeError(f"classify_image() not found in {script_path}")
    return mod["classify_image"]


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS])


def run_batch(model_name: str, classifier_fn, images):
    results = []
    print(f"\n=== Running {model_name} on {len(images)} image(s) ===")
    for idx, img in enumerate(images, start=1):
        try:
            pred = classifier_fn(str(img))
            results.append(
                {
                    "model": model_name,
                    "image": str(img),
                    "prediction": pred if pred is not None else "",
                    "status": "SUCCESS" if pred is not None else "FAILED",
                    "error": "",
                }
            )
            print(f"[{idx}/{len(images)}] OK   {img.name} -> {pred}")
        except Exception as exc:
            results.append(
                {
                    "model": model_name,
                    "image": str(img),
                    "prediction": "",
                    "status": "FAILED",
                    "error": str(exc),
                }
            )
            print(f"[{idx}/{len(images)}] FAIL {img.name} -> {exc}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch inference for ResNet/DenseNet classifiers")
    parser.add_argument("--model", choices=["resnet", "densenet", "both"], default="both")
    parser.add_argument("--input", required=True, help="Image file or folder path")
    parser.add_argument("--output", default="", help="Optional output CSV path")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of input images (0 = all)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    images = collect_images(input_path)
    if not images:
        raise RuntimeError(f"No supported images found under: {input_path}")

    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    all_results = []

    if args.model in {"resnet", "both"}:
        resnet_fn = load_classifier(repo_root / "RESNET50_SINGLEIMAGE.py")
        all_results.extend(run_batch("resnet", resnet_fn, images))

    if args.model in {"densenet", "both"}:
        densenet_fn = load_classifier(repo_root / "DENSENET121_SINGLEIMAGE")
        all_results.extend(run_batch("densenet", densenet_fn, images))

    default_out = (
        repo_root
        / "outputs"
        / "predictions"
        / f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    output_csv = Path(args.output) if args.output else default_out
    if not output_csv.is_absolute():
        output_csv = repo_root / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "image", "prediction", "status", "error"])
        writer.writeheader()
        writer.writerows(all_results)

    success_count = sum(1 for r in all_results if r["status"] == "SUCCESS")
    print("\n=== Batch Inference Summary ===")
    print(f"Input images: {len(images)}")
    print(f"Total rows : {len(all_results)}")
    print(f"Success    : {success_count}")
    print(f"Failed     : {len(all_results) - success_count}")
    print(f"CSV output : {output_csv}\n")


if __name__ == "__main__":
    main()
