"""Single-model baseline evaluation — sends image + prompt directly to a VLM.

No council, no agents — just one model, one prompt, one answer.
Used as a baseline to compare against the multi-agent council.

Usage:
    # Single image test
    python baseline_eval.py 74bPHM081cMUaNKT_4 --verbose

    # Full batch
    python baseline_eval.py --all --concurrency 1

    # With specific model
    python baseline_eval.py --all --model gemma4:27b

    # Use clean images instead of bboxes
    python baseline_eval.py --all --clean-image-dir clean_images_GeoRC
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import click

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from evaluation.metrics import parse_country_result, haversine_km, geoguessr_score, country_match


SYSTEM_PROMPT = """\
Please analyze the street view step-by-step using the following criteria: \
(1) latitude and longitude, (2) sun position, (3) vegetation, (4) natural scenery, \
(5) buildings, (6) license plates, (7) road directions, (8) flags, (9) language, \
(10) shops, and (11) pedestrians. Provide a detailed analysis based on these features. \
Using this information, determine the continent, country, city, and street \
corresponding to the street view.\
"""

USER_PROMPT = """\
The location names should be provided in English. Avoid special characters in your response. \
Think carefully and consider alternatives, but once you've weighed the evidence, commit to your best answer. \
You MUST end your response with the JSON output. \
Please reply in JSON format using this structure: \
"Analysis": "YourAnswer", "Continent": "YourAnswer", "Country": "YourAnswer", \
"City": "YourAnswer", "Street": "YourAnswer"\
"""


@dataclass
class BaselineResult:
    location_id: str
    raw_response: str
    predicted_country: str
    predicted_city: str
    predicted_continent: str
    gt_country: str
    gt_lat: float
    gt_lng: float
    pred_lat: float
    pred_lon: float
    dist_km: float
    geoguessr_score: int
    is_country_match: bool
    error: str = ""


def _encode_image(path: str) -> tuple[str, str]:
    p = Path(path)
    mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_types.get(p.suffix.lower(), "image/jpeg")
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def _parse_baseline_response(text: str) -> dict:
    """Parse the JSON response from the baseline model."""
    # Strip thinking tags if present (Gemma 4 thinking mode)
    clean = text
    if '<|channel>' in clean:
        clean = re.sub(r'<\|channel\>thought\n.*?<channel\|>', '', clean, flags=re.DOTALL)

    # Try to extract JSON from the cleaned response
    json_match = re.search(r'\{[^{}]*"Country"[^{}]*\}', clean, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to find key-value pairs in cleaned text
    result = {}
    for key in ["Country", "City", "Continent", "Street", "Analysis"]:
        pattern = rf'"{key}"\s*:\s*"([^"]*)"'
        m = re.search(pattern, clean, re.IGNORECASE)
        if m:
            result[key] = m.group(1)

    # Fallback: extract from thinking text if no JSON found
    if not result.get("Country"):
        # Look for "Country: X" patterns in the raw thinking
        m = re.search(r'Country:\s*([A-Z][a-zA-Z\s\-]+?)(?:\n|$|,|\.|;)', text)
        if m:
            result["Country"] = m.group(1).strip()
        # Look for "Continent: X" / "City: X" too
        for key in ["Continent", "City"]:
            m = re.search(rf'{key}:\s*([A-Z][a-zA-Z\s\-]+?)(?:\n|$|,|\.|;)', text)
            if m:
                result[key] = m.group(1).strip()

    return result


def _geocode_city_country(city: str, country: str) -> tuple[float, float]:
    """Simple geocoding using the judge's geocode tool logic."""
    try:
        from council.tools import geocode
        result = asyncio.get_event_loop().run_until_complete(
            geocode.ainvoke(f"{city}, {country}")
        )
        # Parse "lat, lon" from result
        if "," in str(result):
            parts = str(result).split(",")
            return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        pass

    # Fallback: just geocode the country
    try:
        from council.tools import geocode
        result = asyncio.get_event_loop().run_until_complete(
            geocode.ainvoke(country)
        )
        if "," in str(result):
            parts = str(result).split(",")
            return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        pass

    return float("nan"), float("nan")


async def run_single_baseline(
    image_path: str,
    location_id: str,
    gt_country: str,
    gt_lat: float,
    gt_lng: float,
    model: str,
    verbose: bool = False,
) -> BaselineResult:
    """Send one image to the VLM and parse the response."""

    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    vllm_url = os.environ.get("BASELINE_VLLM_URL", "")

    b64, mime = _encode_image(image_path)

    if vllm_url:
        # vLLM OpenAI-compatible endpoint
        import httpx

        thinking = os.environ.get("BASELINE_THINKING", "false").lower() == "true"
        b64_url = f"data:{mime};base64,{b64}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": b64_url}},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ],
            "max_tokens": 8192,
            "temperature": 0,
        }

        if thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
            payload["skip_special_tokens"] = False

        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(f"{vllm_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]["message"]
        raw = choice.get("content", "")
        reasoning = choice.get("reasoning", "")
        if reasoning and verbose:
            print(f"  [{location_id}] Thinking: {reasoning[:200]}...", flush=True)
    elif provider == "hai":
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        base_url = os.environ.get("HAI_BASE_URL", "http://localhost:6655/anthropic/")
        api_key = os.environ.get("HAI_API_KEY", "")
        llm = ChatAnthropic(model=model, base_url=base_url, api_key=api_key,
                            temperature=0, max_tokens=4096)

        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]),
        ])
        raw = response.content
    else:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        llm = ChatOllama(model=model, temperature=0, base_url=host, num_predict=4096)

        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]),
        ])
        raw = response.content

    if verbose:
        print(f"  [{location_id}] Raw response: {raw[:200]}", flush=True)

    # Parse response
    parsed = _parse_baseline_response(raw)
    pred_country = parsed.get("Country", "")
    pred_city = parsed.get("City", "")
    pred_continent = parsed.get("Continent", "")

    # Geocode prediction
    pred_lat, pred_lon = _geocode_city_country(pred_city, pred_country)

    # Calculate metrics
    if not math.isnan(pred_lat) and not math.isnan(gt_lat):
        dist = haversine_km(pred_lat, pred_lon, gt_lat, gt_lng)
        score = geoguessr_score(dist)
    else:
        dist = float("nan")
        score = 0

    is_match = country_match(pred_country, gt_country) if pred_country and gt_country else False

    return BaselineResult(
        location_id=location_id,
        raw_response=raw,
        predicted_country=pred_country,
        predicted_city=pred_city,
        predicted_continent=pred_continent,
        gt_country=gt_country,
        gt_lat=gt_lat,
        gt_lng=gt_lng,
        pred_lat=pred_lat,
        pred_lon=pred_lon,
        dist_km=dist,
        geoguessr_score=score,
        is_country_match=is_match,
    )


async def run_batch(
    results_dir: Path,
    model: str,
    output_name: str,
    clean_image_dir: Path | None,
    concurrency: int,
    verbose: bool,
    location_id: str | None = None,
) -> list[BaselineResult]:
    """Run baseline eval on all images."""

    # Load ground truth
    gt_map = {}
    for csv_file in ["Images/georc_locations.csv", "georc_locations.csv"]:
        p = Path(csv_file)
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                loc = row["filename"].replace(".png", "")
                gt_map[loc] = {
                    "lat": float(row["lat"]),
                    "lng": float(row["lng"]),
                    "country": row.get("country_code", row.get("country", "")),
                }

    # Collect images
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    tasks = []
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        loc_id = folder.name
        if location_id and loc_id != location_id:
            continue

        # Skip if already done
        out_path = folder / output_name
        if out_path.exists():
            try:
                data = json.loads(out_path.read_text())
                if data.get("predicted_country") and not data.get("error"):
                    if verbose:
                        print(f"  [{loc_id}] cached", flush=True)
                    continue
            except Exception:
                pass

        # Find image
        if clean_image_dir:
            candidates = [clean_image_dir / f"{loc_id}{ext}" for ext in IMAGE_EXTS]
            image_path = next((c for c in candidates if c.exists()), None)
        else:
            image_path = folder / "bboxes.png"
            if not image_path.exists():
                for ext in IMAGE_EXTS:
                    candidate = folder / f"bboxes{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break

        if not image_path or not image_path.exists():
            if verbose:
                print(f"  [{loc_id}] no image found, skipping", flush=True)
            continue

        gt = gt_map.get(loc_id, {})
        tasks.append((str(image_path), loc_id, gt.get("country", ""),
                      gt.get("lat", float("nan")), gt.get("lng", float("nan"))))

    print(f"Images to process: {len(tasks)}")

    # Run with concurrency
    sem = asyncio.Semaphore(concurrency)
    results = []
    done = 0

    async def bounded(args):
        nonlocal done
        async with sem:
            img_path, loc_id, gt_c, gt_lat, gt_lng = args
            t0 = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    run_single_baseline(img_path, loc_id, gt_c, gt_lat, gt_lng, model, verbose),
                    timeout=600,
                )
            except Exception as e:
                result = BaselineResult(
                    location_id=loc_id, raw_response="", predicted_country="",
                    predicted_city="", predicted_continent="", gt_country=gt_c,
                    gt_lat=gt_lat, gt_lng=gt_lng, pred_lat=float("nan"),
                    pred_lon=float("nan"), dist_km=float("nan"), geoguessr_score=0,
                    is_country_match=False, error=str(e),
                )
            elapsed = time.monotonic() - t0
            done += 1

            # Save result
            out_path = results_dir / loc_id / output_name
            out_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))

            status = "OK" if not result.error else "ERR"
            country_str = result.predicted_country or "?"
            print(f"  [{status}] [{done}/{len(tasks)}] {loc_id} -> {country_str} ({elapsed:.0f}s)", flush=True)

            return result

    all_results = await asyncio.gather(*[bounded(t) for t in tasks])
    return list(all_results)


@click.command()
@click.argument("location_id", required=False, default=None)
@click.option("--all", "run_all", is_flag=True, help="Run on all images")
@click.option("--model", default="gemma4:31b", show_default=True, help="Vision model to use")
@click.option("--results-dir", default="results GeoRC", type=click.Path(file_okay=False))
@click.option("--output-name", default="baseline_gemma4_result.json", show_default=True)
@click.option("--clean-image-dir", default=None, type=click.Path(file_okay=False))
@click.option("--concurrency", default=1, type=int)
@click.option("--verbose", is_flag=True)
def main(location_id, run_all, model, results_dir, output_name, clean_image_dir, concurrency, verbose):
    if not location_id and not run_all:
        click.echo("Provide a LOCATION_ID or use --all")
        return

    results_path = Path(results_dir)
    clean_path = Path(clean_image_dir) if clean_image_dir else None

    print(f"Model: {model}")
    print(f"Provider: {os.environ.get('LLM_PROVIDER', 'ollama')}")
    print(f"Output: {output_name}")

    results = asyncio.run(run_batch(
        results_path, model, output_name, clean_path, concurrency, verbose, location_id
    ))

    if results:
        n_match = sum(1 for r in results if r.is_country_match)
        valid_scores = [r.geoguessr_score for r in results if not math.isnan(r.dist_km)]
        valid_dists = [r.dist_km for r in results if not math.isnan(r.dist_km)]
        print(f"\n{'='*50}")
        print(f"Country match: {n_match}/{len(results)} ({n_match/len(results)*100:.1f}%)")
        if valid_scores:
            print(f"Mean GeoGuessr score: {sum(valid_scores)/len(valid_scores):.0f}")
            print(f"Mean distance: {sum(valid_dists)/len(valid_dists):.0f} km")


if __name__ == "__main__":
    main()
