#!/usr/bin/env python3
import sys
import os
import csv
import json
import argparse
import base64
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re

import requests
import pandas as pd
from slugify import slugify
import os as _os
import time

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# Replicate (optional)
try:
    import replicate as _replicate
    _REPLICATE_AVAILABLE = True
except Exception:
    _REPLICATE_AVAILABLE = False

# --- Constants guided by Amazon SEO sources (see README) ---
TITLE_MAX = 200  # hard cap; we'll aim ~60-120
BACKEND_BYTES_TARGET = 240  # close to 249 bytes, but leave margin

# --- Lightweight .env loader (no extra dependency) ---
def _load_env_from_dotenv():
    paths = [
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.getcwd(), '.env'),
    ]
    for p in paths:
        try:
            if not os.path.exists(p):
                continue
            with open(p, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and (k not in os.environ or not os.environ.get(k)):
                        os.environ[k] = v
        except Exception:
            # ignore dotenv issues silently
            pass
# --- Data models ---
@dataclass
class ArtStyle:
    style: str
    description: str
    example_tags: str

@dataclass
class Concept:
    niche: str
    style: str
    image_prompt: str
    primary_keywords: List[str]
    secondary_keywords: List[str]
    title: str
    description: str
    backend_keywords: str  # space-separated; try to fit ~249 bytes
    image_path: str = ""   # optional path to generated image

# --- Helpers ---

def read_art_styles(path: str) -> List[ArtStyle]:
    styles: List[ArtStyle] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            styles.append(ArtStyle(
                style=row['style'].strip(),
                description=row['description'].strip(),
                example_tags=row['example_tags'].strip(),
            ))
    return styles


def expand_keywords(niche: str) -> Tuple[List[str], List[str]]:
    """Heuristic keyword expansion: returns (primary, secondary).
    Mix of short-tail and long-tail variants. No brand names or forbidden terms.
    """
    n = niche.lower().strip()
    core = [n]
    tokens = [t for t in n.replace('-', ' ').split() if t]

    # Short-tail cores
    short_tail = list(dict.fromkeys(core + tokens))

    # Long-tail templates
    long_tail_templates = [
        f"{n} t-shirt",
        f"{n} tee",
        f"{n} shirt",
        f"{n} graphic tee",
        f"{n} gift",
        f"{n} aesthetic shirt",
    ]
    # Avoid redundant 'funny funny ...' when niche already contains 'funny'
    if 'funny' not in n:
        long_tail_templates.append(f"funny {n} shirt")
    long_tail_templates += [
        f"vintage {n} tee",
        f"{n} for men",
        f"{n} for women",
        f"{n} for dad",
        f"{n} for mom",
    ]

    # De-duplicate, keep order
    long_tail = list(dict.fromkeys([t for t in long_tail_templates if t != n]))

    # Primary: top 3-5 concise; Secondary: broader and long-tail
    primary = short_tail[:5]
    secondary = long_tail + short_tail[5:]
    return primary, secondary


def choose_styles(all_styles: List[ArtStyle], niche: str, k: int = 4) -> List[ArtStyle]:
    # Simple diversity pick by category keywords and spread across file
    priority_buckets = [
        'Vintage/Retro', 'Minimalist', 'Anime', 'Street', 'Tattoo', 'Nature', 'Typography', 'Retro Futurism',
        'Watercolor', 'Comic', 'Geometric', 'Grunge', 'Stencil', 'Pixel', 'Kawaii', 'Gothic', 'Art Deco', 'Art Nouveau'
    ]
    picked: List[ArtStyle] = []

    # First, try to map niche hints
    n = niche.lower()
    hints = [
        ('vintage', 'Vintage/Retro'),
        ('retro', 'Vintage/Retro'),
        ('minimal', 'Minimalist Line Art'),
        ('anime', 'Anime/Manga'),
        ('manga', 'Anime/Manga'),
        ('street', 'Streetwear/Graffiti'),
        ('tattoo', 'Tattoo/Old School'),
        ('nature', 'Nature Illustration'),
        ('botanical', 'Nature Illustration'),
        ('type', 'Typography-Only'),
        ('synthwave', 'Retro Futurism'),
        ('watercolor', 'Watercolor'),
        ('comic', 'Comic Book Pop Art'),
        ('geometric', 'Geometric/Abstract'),
        ('grunge', 'Grunge/Distressed'),
        ('stencil', 'Stencil/Silhouette'),
        ('pixel', 'Pixel/8-bit'),
        ('kawaii', 'Kawaii/Cute'),
        ('goth', 'Gothic/Dark Aesthetic'),
        ('deco', 'Art Deco'),
        ('nouveau', 'Art Nouveau'),
    ]
    hinted_names = [name for key, name in hints if key in n]
    for hn in hinted_names:
        for s in all_styles:
            if hn.lower() in s.style.lower():
                if s not in picked:
                    picked.append(s)
                    if len(picked) >= k:
                        return picked

    # Then, fill from priority buckets if exist
    for pb in priority_buckets:
        for s in all_styles:
            if pb.lower() in s.style.lower() and s not in picked:
                picked.append(s)
                if len(picked) >= k:
                    return picked

    # Fallback: first k styles
    if len(picked) < k:
        for s in all_styles:
            if s not in picked:
                picked.append(s)
            if len(picked) >= k:
                break
    return picked[:k]


def build_image_prompt(niche: str, style: ArtStyle) -> str:
    # Prompt shaped for general image generators; avoids brand/IP; mentions print constraints
    tags = style.example_tags.replace(',', ' ')
    quoted = re.findall(r'"([^"]+)"', niche)
    phrase_hint = ""
    if quoted:
        # Encourage exact text rendering with high legibility and minimal extra text
        q = quoted[0].strip()
        if q:
            phrase_hint = (
                f" Include the exact phrase \"{q}\" prominently; correct spelling as provided; high legibility; "
                f"minimal or no additional text; good kerning; thick strokes.")
    prompt = (
        f"{niche} in {style.style} style, {style.description} — {tags}. "
        f"Center composition for t-shirt screen-print, clean silhouette, vector-friendly edges, high contrast, limited colors." + phrase_hint
    )
    return prompt


def clamp_text(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len-1].rstrip() + '…'


def make_title(niche: str, primary_keywords: List[str], style: ArtStyle) -> str:
    # Include primary keyword + style descriptor + product type
    pk = primary_keywords[0] if primary_keywords else niche
    base = f"{pk.title()} {style.style} Graphic T-Shirt"
    return clamp_text(base, TITLE_MAX)


def make_description(niche: str, primary: List[str], secondary: List[str], style: ArtStyle) -> str:
    # Natural-language paragraph using keyword variations once; include materials/care placeholders
    key_phrase = primary[0] if primary else niche
    sec = ', '.join(list(dict.fromkeys(secondary[:8])))
    text = (
        f"Express your style with this {style.style.lower()} {key_phrase} graphic tee. "
        f"Designed for everyday comfort with soft cotton feel and a classic fit. "
        f"Inspired by {style.description.lower()} for a unique look. "
        f"Great gift idea for fans of {niche}. Keywords: {sec}. "
        f"Care: machine wash cold, inside-out; tumble dry low."
    )
    return clamp_text(text, 1000)


def make_backend_keywords(primary: List[str], secondary: List[str]) -> str:
    # Space-separated; lowercase; no commas; avoid repeats; aim near byte limit
    terms = list(dict.fromkeys([*(p.lower() for p in primary), *(s.lower() for s in secondary)]))
    # prune obviously redundant words
    stop = {"shirt", "t-shirt", "tee", "for", "and", "the", "a", "gift"}
    terms = [t for t in terms if t not in stop]

    # pack until approx BACKEND_BYTES_TARGET bytes (utf-8)
    packed_tokens: List[str] = []
    total = 0
    for t in terms:
        b = t.encode('utf-8')
        sep = 1 if packed_tokens else 0
        if total + len(b) > BACKEND_BYTES_TARGET:
            break
        total += len(b) + sep
        packed_tokens.append(t)
    return ' '.join(packed_tokens)


def generate_concepts(niche: str, styles_db: List[ArtStyle]) -> List[Concept]:
    primary, secondary = expand_keywords(niche)
    styles = choose_styles(styles_db, niche, k=4)

    concepts: List[Concept] = []
    for s in styles:
        prompt = build_image_prompt(niche, s)
        title = make_title(niche, primary, s)
        desc = make_description(niche, primary, secondary, s)
        backend = make_backend_keywords(primary, secondary)
        concepts.append(Concept(
            niche=niche,
            style=s.style,
            image_prompt=prompt,
            primary_keywords=primary,
            secondary_keywords=secondary,
            title=title,
            description=desc,
            backend_keywords=backend,
        ))
    return concepts


def generate_images_a1111(sd_url: str, concepts: List[Concept], niche: str, steps: int = 28, cfg: float = 6.5, width: int = 768, height: int = 768) -> None:
    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    for idx, c in enumerate(concepts, start=1):
        try:
            payload = {
                "prompt": c.image_prompt,
                # Allow text so quoted phrases can render; still avoid low quality artifacts
                "negative_prompt": "low-res, blurry, jpeg artifacts, watermark, logo, copyright, trademark, misspelling, typos",
                "steps": steps,
                "cfg_scale": cfg,
                "width": width,
                "height": height,
            }
            r = requests.post(f"{sd_url}/sdapi/v1/txt2img", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            if 'images' in data and data['images']:
                b64 = data['images'][0]
                if ',' in b64:
                    b64 = b64.split(',', 1)[-1]
                img_bytes = base64.b64decode(b64)
                out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
                with open(out_path, 'wb') as f:
                    f.write(img_bytes)
                c.image_path = out_path
        except Exception as e:
            print(f"[warn] image generation failed for #{idx}: {e}")


def generate_images_openai(concepts: List[Concept], niche: str, model: str = "gpt-image-1", size: str = "1024x1024") -> None:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not available; install openai and set OPENAI_API_KEY")
    # Support multiple keys: CLI (--openai-keys) not available here, so read from env OPENAI_API_KEYS (comma-separated) and fallback to OPENAI_API_KEY
    keys_raw = _os.environ.get("OPENAI_API_KEYS") or ""
    keys: List[str] = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        single = _os.environ.get("OPENAI_API_KEY")
        if not single:
            raise RuntimeError("Set OPENAI_API_KEY or OPENAI_API_KEYS in environment")
        keys = [single]
    # Map alias 'latest' to a concrete model name via env or default
    if (model or "").lower() == "latest":
        model = _os.environ.get("OPENAI_IMAGE_MODEL_LATEST", "gpt-image-1")

    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    for idx, c in enumerate(concepts, start=1):
        last_err: Exception | None = None
        for api_key in keys:
            try:
                client = OpenAI(api_key=api_key)
                # OpenAI Images API: create image from prompt
                result = client.images.generate(model=model, prompt=c.image_prompt, size=size)
                # API returns base64 in data[0].b64_json
                b64 = result.data[0].b64_json
                img_bytes = base64.b64decode(b64)
                out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
                with open(out_path, 'wb') as f:
                    f.write(img_bytes)
                c.image_path = out_path
                last_err = None
                break
            except Exception as e:
                last_err = e
                # Try next key if available (useful for billing limits or per-key rate limits)
                continue
        if last_err is not None and not c.image_path:
            print(f"[warn] openai image generation failed for #{idx} across all keys: {last_err}")


def _find_b64_values(obj) -> List[str]:
    vals: List[str] = []
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'b64_json' and isinstance(v, str):
                    vals.append(v)
                else:
                    vals.extend(_find_b64_values(v))
        elif isinstance(obj, list):
            for v in obj:
                vals.extend(_find_b64_values(v))
    except Exception:
        pass
    return vals


def generate_images_openai_streaming(concepts: List[Concept], niche: str, model: str = "gpt-image-1", size: str = "1024x1024") -> None:
    """Generate images using OpenAI Images Streaming API via Python SDK.
    Requires openai>=1.2x with images.with_streaming_response.
    It scans streamed events for any 'b64_json' payloads and writes the first image found per concept.
    Falls back across multiple keys via OPENAI_API_KEYS if provided.
    """
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not available; install openai and set OPENAI_API_KEY")
    keys_raw = _os.environ.get("OPENAI_API_KEYS") or ""
    keys: List[str] = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        single = _os.environ.get("OPENAI_API_KEY")
        if not single:
            raise RuntimeError("Set OPENAI_API_KEY or OPENAI_API_KEYS in environment")
        keys = [single]
    if (model or "").lower() == "latest":
        model = _os.environ.get("OPENAI_IMAGE_MODEL_LATEST", "gpt-image-1")

    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    from openai import OpenAI

    for idx, c in enumerate(concepts, start=1):
        last_err: Exception | None = None
        for api_key in keys:
            try:
                client = OpenAI(api_key=api_key)
                # Use streaming response wrapper; request b64_json format to simplify extraction
                with client.images.with_streaming_response.generate(
                    model=model,
                    prompt=c.image_prompt,
                    size=size,
                    response_format="b64_json",
                ) as response:
                    b64_collected: List[str] = []
                    try:
                        # Iterate SDK stream events (SSE). Each event is typically a dict-like chunk.
                        stream = response.parse()
                        for event in stream:
                            # Attempt to discover base64 payloads within this event structure
                            b64s = []
                            if isinstance(event, dict):
                                # Events may have shape {"data": {...}, "event": "..."}
                                b64s.extend(_find_b64_values(event))
                                data = event.get("data")
                                if data is not None:
                                    b64s.extend(_find_b64_values(data))
                            else:
                                # Fallback: try attribute access or direct value
                                try:
                                    data = getattr(event, "data", None)
                                    if data is not None:
                                        b64s.extend(_find_b64_values(data))
                                except Exception:
                                    pass
                            if b64s:
                                b64_collected.extend(b64s)
                    except Exception as stream_err:
                        last_err = stream_err
                        continue

                # If any base64 found, write first; if multiple parts, attempt to join
                if b64_collected:
                    b64_joined = "".join(b64_collected)
                    try:
                        img_bytes = base64.b64decode(b64_joined)
                    except Exception:
                        # If concatenation fails, try first element only
                        img_bytes = base64.b64decode(b64_collected[0])
                    out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
                    with open(out_path, 'wb') as f:
                        f.write(img_bytes)
                    c.image_path = out_path
                    last_err = None
                    break
                else:
                    last_err = RuntimeError("No image data received in stream events")
                    continue
            except Exception as e:
                last_err = e
                continue
        if last_err is not None and not c.image_path:
            print(f"[warn] openai streaming failed for #{idx} across all keys: {last_err}")


def _resolve_gemini_model(name: str) -> str:
    """Map friendly/legacy names to supported Google image generation model.
    For image generation, prefer Imagen 3 models via Generative Language API.
    """
    if not name:
        return "imagen-3.0"
    n = name.strip().lower()
    # If a gemini text model was supplied, prefer an Imagen model for images.
    if n.startswith("gemini"):
        return "imagen-3.0"
    # Accept imagen aliases
    if n in {"imagen", "imagen-3", "imagen-3.0", "imagen-3.0-fast", "imagen-3.0-light"} or n.startswith("imagen"):
        return name
    # Fallback to imagen
    return "imagen-3.0"


def _extract_inline_b64_from_gemini_response_obj(resp) -> str | None:
    """Attempt to extract base64 image from various SDK response shapes."""
    try:
        # SDK objects: resp.candidates[0].content.parts[0].inline_data.data
        candidates = getattr(resp, 'candidates', None)
        if candidates:
            cand0 = candidates[0]
            content = getattr(cand0, 'content', None) or getattr(cand0, 'content', {})
            parts = getattr(content, 'parts', None) or getattr(content, 'parts', [])
            if parts:
                p0 = parts[0]
                inline = getattr(p0, 'inline_data', None) or {}
                data = getattr(inline, 'data', None)
                if data:
                    return data
        # Dict-like
        if isinstance(resp, dict):
            candidates = resp.get('candidates') or []
            if candidates:
                parts = ((candidates[0] or {}).get('content') or {}).get('parts') or []
                if parts and isinstance(parts, list):
                    inline = parts[0].get('inlineData') or {}
                    data = inline.get('data')
                    if data:
                        return data
    except Exception:
        pass
    return None


def _generate_image_gemini_single(prompt: str, out_path: str, model: str, api_key: str) -> str:
    """Try SDK first, then REST fallback. Returns out_path on success or raises."""
    model = _resolve_gemini_model(model)

    # Try SDK if available
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        # Tools specification differs across SDK versions; try permissive dict first
        try:
            gm = genai.GenerativeModel(model, tools=[{"image_generation": {}}])
        except TypeError:
            # Older versions expect model_name kwarg
            gm = genai.GenerativeModel(model_name=model, tools=[{"image_generation": {}}])
        # Request PNG inline
        resp = gm.generate_content(prompt, generation_config={"response_mime_type": "image/png"})
        b64 = _extract_inline_b64_from_gemini_response_obj(resp)
        if b64:
            _ensure_parent(out_path)
            with open(out_path, 'wb') as f:
                f.write(base64.b64decode(b64))
            return out_path
    except Exception as sdk_err:
        # Continue to REST fallback, but keep context if needed
        last_sdk_err = sdk_err  # noqa: F841

    # REST fallback
    try:
        base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"image_generation": {}}],
            "generationConfig": {"responseMimeType": "image/png"}
        }
        r = requests.post(base_url, json=payload, timeout=120)
        if not r.ok:
            # Surface server message for troubleshooting
            try:
                msg = r.text
            except Exception:
                msg = str(r.status_code)
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {msg}")
        data = r.json()
        b64 = _extract_inline_b64_from_gemini_response_obj(data)
        if not b64:
            raise RuntimeError("Gemini response missing inline image data")
        _ensure_parent(out_path)
        with open(out_path, 'wb') as f:
            f.write(base64.b64decode(b64))
        return out_path
    except Exception as rest_err:
        raise rest_err


def generate_images_gemini(concepts: List[Concept], niche: str, model: str = "gemini-1.5-flash", aspect: str = "1:1") -> None:
    """Generate images via Gemini using SDK first with REST fallback. Requires GEMINI_API_KEY."""
    api_key = _os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment")

    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    for idx, c in enumerate(concepts, start=1):
        try:
            out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
            p = _generate_image_gemini_single(c.image_prompt, out_path, model, api_key)
            c.image_path = p
        except Exception as e:
            # Include brief message for 400s
            print(f"[warn] gemini image generation failed for #{idx}: {e}")
    


# --- Module-scope helpers for provider order (OpenAI support) ---
def _ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _generate_image_openai_single(prompt: str, out_path: str, model: str, size: str, api_key: str):
    if not _OPENAI_AVAILABLE:
        raise RuntimeError('openai package not available')
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    if (model or "").lower() == "latest":
        model = _os.environ.get("OPENAI_IMAGE_MODEL_LATEST", "gpt-image-1")
    _ensure_parent(out_path)
    resp = client.images.generate(model=model, prompt=prompt, size=size)
    b64 = resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    with open(out_path, 'wb') as f:
        f.write(img_bytes)
    return out_path


def generate_images_failover(concepts: List[Concept], niche: str, args, provider_order: List[str]) -> None:
    """Minimal failover supporting OpenAI and a1111 to satisfy --provider-order usage."""
    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    OPENAI_KEY = _os.environ.get('OPENAI_API_KEY')
    GEMINI_KEY = _os.environ.get('GEMINI_API_KEY')

    for idx, c in enumerate(concepts, start=1):
        prompt = c.image_prompt
        out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
        success = False
        last_err = None

        for provider in provider_order:
            try:
                if provider == 'openai' and OPENAI_KEY:
                    p = _generate_image_openai_single(prompt, out_path, getattr(args, 'openai_model', 'gpt-image-1'), getattr(args, 'openai_size', '1024x1024'), OPENAI_KEY)
                    c.image_path = p
                    success = True
                    break
                elif provider == 'a1111':
                    payload = {
                        "prompt": prompt,
                        "negative_prompt": "low-res, blurry, jpeg artifacts, watermark, logo, copyright, trademark, misspelling, typos",
                        "steps": getattr(args, 'steps', 28),
                        "cfg_scale": getattr(args, 'cfg', 6.5),
                        "width": getattr(args, 'width', 768),
                        "height": getattr(args, 'height', 768),
                    }
                    r = requests.post(f"{args.sd_url.rstrip('/')}/sdapi/v1/txt2img", json=payload, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    images = data.get('images') or []
                    if not images:
                        raise RuntimeError('A1111 returned no images')
                    b64s = images[0]
                    if ',' in b64s:
                        b64s = b64s.split(',', 1)[-1]
                    img_bytes = base64.b64decode(b64s)
                    with open(out_path, 'wb') as f:
                        f.write(img_bytes)
                    c.image_path = out_path
                    success = True
                    break
                elif provider == 'gemini' and GEMINI_KEY:
                    try:
                        gm = getattr(args, 'gemini_model', 'gemini-1.5-flash')
                        p = _generate_image_gemini_single(prompt, out_path, gm, GEMINI_KEY)
                        c.image_path = p
                        success = True
                        break
                    except Exception as e:
                        last_err = e
                        continue
                else:
                    continue
            except Exception as e:
                last_err = e
                continue

        if not success and last_err:
            print(f"[warn] all providers failed for image #{idx}: {last_err}")


def generate_images_replicate(concepts: List[Concept], niche: str, model: str = "ideogram-ai/ideogram-v3-turbo", aspect: str = "1:1", token: str = "") -> None:
    """Generate images via Replicate. Defaults to Ideogram v3 Turbo model.
    - Reads API token from REPLICATE_API_TOKEN unless provided explicitly via token.
    - Saves the first returned image per concept.
    """
    if not _REPLICATE_AVAILABLE:
        raise RuntimeError("replicate package not available; install replicate and set REPLICATE_API_TOKEN")
    api_token = token or _os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise RuntimeError("REPLICATE_API_TOKEN is not set; pass --replicate-token or export env var")

    # Replicate's Python client reads the token from env var
    _os.environ["REPLICATE_API_TOKEN"] = api_token

    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    for idx, c in enumerate(concepts, start=1):
        # Respect free-tier rate limits: ~6/min with burst=1
        if idx > 1:
            time.sleep(10)
        attempts = 0
        while attempts < 3 and not c.image_path:
            attempts += 1
            try:
                output = _replicate.run(model, input={
                    "prompt": c.image_prompt,
                    "aspect_ratio": aspect,
                })
                urls: List[str] = []
                if isinstance(output, list):
                    urls = [u for u in output if isinstance(u, str)]
                elif isinstance(output, dict) and 'output' in output:
                    out = output['output']
                    if isinstance(out, list):
                        urls = [u for u in out if isinstance(u, str)]
                if urls:
                    img_url = urls[0]
                    r = requests.get(img_url, timeout=120)
                    r.raise_for_status()
                    out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
                    with open(out_path, 'wb') as f:
                        f.write(r.content)
                    c.image_path = out_path
                else:
                    if attempts < 3:
                        print(f"[warn] replicate output empty for #{idx}, retrying...")
                        time.sleep(4)
                    else:
                        print(f"[warn] replicate output empty for #{idx} after retries")
            except Exception as e:
                msg = str(e)
                if "429" in msg or "throttled" in msg.lower():
                    wait_s = 6 if attempts == 1 else 12
                    print(f"[warn] replicate rate limited for #{idx}, waiting {wait_s}s and retrying...")
                    time.sleep(wait_s)
                    continue
                else:
                    print(f"[warn] replicate image generation failed for #{idx}: {e}")


        # end replicate helper block


def export_outputs(niche: str, concepts: List[Concept], art_styles: List[ArtStyle]):
    os.makedirs('output', exist_ok=True)
    slug = slugify(niche)

    # Excel for concepts
    rows: List[Dict[str, str]] = []
    for c in concepts:
        rows.append({
            'niche': c.niche,
            'style': c.style,
            'title': c.title,
            'description': c.description,
            'backend_keywords': c.backend_keywords,
            'primary_keywords': ', '.join(c.primary_keywords),
            'secondary_keywords': ', '.join(c.secondary_keywords),
            'image_prompt': c.image_prompt,
            'image_path': c.image_path,
        })
    df = pd.DataFrame(rows)
    out_xlsx = os.path.join('output', f'{slug}_listings.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='listings', index=False)
        # Also include art styles on another sheet
        styles_rows = [{'style': s.style, 'description': s.description, 'example_tags': s.example_tags} for s in art_styles]
        df_styles = pd.DataFrame(styles_rows)
        df_styles.to_excel(writer, sheet_name='art_styles', index=False)

    # CSV outputs
    out_csv = os.path.join('output', f'{slug}_listings.csv')
    df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_xlsx}\nWrote: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="POD listing + prompt generator")
    parser.add_argument("niche", type=str, help="Niche description, e.g., 'funny cat dad'")
    parser.add_argument("--images", action="store_true", help="Generate images as well as text outputs")
    parser.add_argument("--provider", type=str, default="a1111", choices=["a1111","openai","replicate","gemini"], help="Image generation provider")
    parser.add_argument("--sd-url", type=str, default="http://127.0.0.1:7860", help="Stable Diffusion WebUI base URL (provider=a1111)")
    parser.add_argument("--steps", type=int, default=28, help="Sampling steps for image generation")
    parser.add_argument("--cfg", type=float, default=6.5, help="CFG scale")
    parser.add_argument("--width", type=int, default=768, help="Image width")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--openai-model", type=str, default="gpt-image-1", help="OpenAI image model name (provider=openai)")
    parser.add_argument("--openai-size", type=str, default="1024x1024", help="OpenAI image size WxH (provider=openai)")
    parser.add_argument("--openai-stream", action="store_true", help="Use OpenAI Images Streaming API (provider=openai)")
    parser.add_argument("--replicate-model", type=str, default="ideogram-ai/ideogram-v3-turbo", help="Replicate model (provider=replicate)")
    parser.add_argument("--replicate-aspect", type=str, default="1:1", help="Replicate aspect ratio, e.g., 1:1, 3:4, 16:9 (provider=replicate)")
    parser.add_argument("--replicate-token", type=str, default="", help="Replicate API token override; else use REPLICATE_API_TOKEN env")
    # Gemini options
    parser.add_argument("--gemini-model", type=str, default="imagen-3.0", help="Google image model id (provider=gemini). Defaults to 'imagen-3.0'.")
    parser.add_argument("--gemini-aspect", type=str, default="1:1", help="Gemini aspect ratio (provider=gemini), e.g., 1:1, 3:4, 16:9")
    # Multi-key rotation and provider failover
    parser.add_argument("--provider-order", type=str, default=None, help="Comma-separated provider order to try, e.g., 'a1111,openai,replicate'")
    parser.add_argument("--openai-keys", type=str, default=None, help="Comma-separated OpenAI API keys for rotation")
    parser.add_argument("--replicate-tokens", type=str, default=None, help="Comma-separated Replicate tokens for rotation")
    # Load .env before parsing args so env-backed defaults can be honored
    _load_env_from_dotenv()
    args = parser.parse_args()

    niche = args.niche.strip()
    styles_path = os.path.join(os.path.dirname(__file__), 'art_styles.csv')
    if not os.path.exists(styles_path):
        print(f"Missing art styles file at {styles_path}")
        sys.exit(2)

    styles = read_art_styles(styles_path)
    concepts = generate_concepts(niche, styles)

    if args.images:
        # If a provider order is given or multiple keys are present, use failover flow; else preserve legacy behavior
        provider_order = None
        if args.provider_order:
            provider_order = [p.strip() for p in args.provider_order.split(',') if p.strip()]
        else:
            # if rotation flags or env lists are present, enable failover starting with selected provider
            has_multi = bool(args.openai_keys or args.replicate_tokens or _os.environ.get('OPENAI_API_KEYS') or _os.environ.get('REPLICATE_API_TOKENS'))
            if has_multi:
                provider_order = [args.provider]

        if provider_order:
            generate_images_failover(concepts, niche, args, provider_order)
        else:
            if args.provider == "a1111":
                generate_images_a1111(args.sd_url, concepts, niche, steps=args.steps, cfg=args.cfg, width=args.width, height=args.height)
            elif args.provider == "openai":
                if args.openai_stream:
                    generate_images_openai_streaming(concepts, niche, model=args.openai_model, size=args.openai_size)
                else:
                    generate_images_openai(concepts, niche, model=args.openai_model, size=args.openai_size)
            elif args.provider == "replicate":
                generate_images_replicate(concepts, niche, model=args.replicate_model, aspect=args.replicate_aspect, token=args.replicate_token)
            elif args.provider == "gemini":
                generate_images_gemini(concepts, niche, model=args.gemini_model, aspect=args.gemini_aspect)

    export_outputs(niche, concepts, styles)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import sys
import os
import csv
import json
import argparse
import base64
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re

import requests
import pandas as pd
from slugify import slugify
import os as _os
import time

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# Replicate (optional)
try:
    import replicate as _replicate
    _REPLICATE_AVAILABLE = True
except Exception:
    _REPLICATE_AVAILABLE = False

# --- Constants guided by Amazon SEO sources (see README) ---
TITLE_MAX = 200  # hard cap; we'll aim ~60-120
BACKEND_BYTES_TARGET = 240  # close to 249 bytes, but leave margin

# --- Lightweight .env loader (no extra dependency) ---
def _load_env_from_dotenv():
    paths = [
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.getcwd(), '.env'),
    ]
    for p in paths:
        try:
            if not os.path.exists(p):
                continue
            with open(p, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and (k not in os.environ or not os.environ.get(k)):
                        os.environ[k] = v
        except Exception:
            # ignore dotenv issues silently
            pass
# --- Data models ---
@dataclass
class ArtStyle:
    style: str
    description: str
    example_tags: str

@dataclass
class Concept:
    niche: str
    style: str
    image_prompt: str
    primary_keywords: List[str]
    secondary_keywords: List[str]
    title: str
    description: str
    backend_keywords: str  # space-separated; try to fit ~249 bytes
    image_path: str = ""   # optional path to generated image

# --- Helpers ---

def read_art_styles(path: str) -> List[ArtStyle]:
    styles: List[ArtStyle] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            styles.append(ArtStyle(
                style=row['style'].strip(),
                description=row['description'].strip(),
                example_tags=row['example_tags'].strip(),
            ))
    return styles


def expand_keywords(niche: str) -> Tuple[List[str], List[str]]:
    """Heuristic keyword expansion: returns (primary, secondary).
    Mix of short-tail and long-tail variants. No brand names or forbidden terms.
    """
    n = niche.lower().strip()
    core = [n]
    tokens = [t for t in n.replace('-', ' ').split() if t]

    # Short-tail cores
    short_tail = list(dict.fromkeys(core + tokens))

    # Long-tail templates
    long_tail_templates = [
        f"{n} t-shirt",
        f"{n} tee",
        f"{n} shirt",
        f"{n} graphic tee",
        f"{n} gift",
        f"{n} aesthetic shirt",
    ]
    # Avoid redundant 'funny funny ...' when niche already contains 'funny'
    if 'funny' not in n:
        long_tail_templates.append(f"funny {n} shirt")
    long_tail_templates += [
        f"vintage {n} tee",
        f"{n} for men",
        f"{n} for women",
        f"{n} for dad",
        f"{n} for mom",
    ]

    # De-duplicate, keep order
    long_tail = list(dict.fromkeys([t for t in long_tail_templates if t != n]))

    # Primary: top 3-5 concise; Secondary: broader and long-tail
    primary = short_tail[:5]
    secondary = long_tail + short_tail[5:]
    return primary, secondary


def choose_styles(all_styles: List[ArtStyle], niche: str, k: int = 4) -> List[ArtStyle]:
    # Simple diversity pick by category keywords and spread across file
    priority_buckets = [
        'Vintage/Retro', 'Minimalist', 'Anime', 'Street', 'Tattoo', 'Nature', 'Typography', 'Retro Futurism',
        'Watercolor', 'Comic', 'Geometric', 'Grunge', 'Stencil', 'Pixel', 'Kawaii', 'Gothic', 'Art Deco', 'Art Nouveau'
    ]
    picked: List[ArtStyle] = []

    # First, try to map niche hints
    n = niche.lower()
    hints = [
        ('vintage', 'Vintage/Retro'),
        ('retro', 'Vintage/Retro'),
        ('minimal', 'Minimalist Line Art'),
        ('anime', 'Anime/Manga'),
        ('manga', 'Anime/Manga'),
        ('street', 'Streetwear/Graffiti'),
        ('tattoo', 'Tattoo/Old School'),
        ('nature', 'Nature Illustration'),
        ('botanical', 'Nature Illustration'),
        ('type', 'Typography-Only'),
        ('synthwave', 'Retro Futurism'),
        ('watercolor', 'Watercolor'),
        ('comic', 'Comic Book Pop Art'),
        ('geometric', 'Geometric/Abstract'),
        ('grunge', 'Grunge/Distressed'),
        ('stencil', 'Stencil/Silhouette'),
        ('pixel', 'Pixel/8-bit'),
        ('kawaii', 'Kawaii/Cute'),
        ('goth', 'Gothic/Dark Aesthetic'),
        ('deco', 'Art Deco'),
        ('nouveau', 'Art Nouveau'),
    ]
    hinted_names = [name for key, name in hints if key in n]
    for hn in hinted_names:
        for s in all_styles:
            if hn.lower() in s.style.lower():
                if s not in picked:
                    picked.append(s)
                    if len(picked) >= k:
                        return picked

    # Then, fill from priority buckets if exist
    for pb in priority_buckets:
        for s in all_styles:
            if pb.lower() in s.style.lower() and s not in picked:
                picked.append(s)
                if len(picked) >= k:
                    return picked

    # Fallback: first k styles
    if len(picked) < k:
        for s in all_styles:
            if s not in picked:
                picked.append(s)
            if len(picked) >= k:
                break
    return picked[:k]


def build_image_prompt(niche: str, style: ArtStyle) -> str:
    # Prompt shaped for general image generators; avoids brand/IP; mentions print constraints
    tags = style.example_tags.replace(',', ' ')
    quoted = re.findall(r'"([^"]+)"', niche)
    phrase_hint = ""
    if quoted:
        # Encourage exact text rendering with high legibility and minimal extra text
        q = quoted[0].strip()
        if q:
            phrase_hint = (
                f" Include the exact phrase \"{q}\" prominently; correct spelling as provided; high legibility; "
                f"minimal or no additional text; good kerning; thick strokes.")
    prompt = (
        f"{niche} in {style.style} style, {style.description} — {tags}. "
        f"Center composition for t-shirt screen-print, clean silhouette, vector-friendly edges, high contrast, limited colors." + phrase_hint
    )
    return prompt


def clamp_text(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len-1].rstrip() + '…'


def make_title(niche: str, primary_keywords: List[str], style: ArtStyle) -> str:
    # Include primary keyword + style descriptor + product type
    pk = primary_keywords[0] if primary_keywords else niche
    base = f"{pk.title()} {style.style} Graphic T-Shirt"
    return clamp_text(base, TITLE_MAX)


def make_description(niche: str, primary: List[str], secondary: List[str], style: ArtStyle) -> str:
    # Natural-language paragraph using keyword variations once; include materials/care placeholders
    key_phrase = primary[0] if primary else niche
    sec = ', '.join(list(dict.fromkeys(secondary[:8])))
    text = (
        f"Express your style with this {style.style.lower()} {key_phrase} graphic tee. "
        f"Designed for everyday comfort with soft cotton feel and a classic fit. "
        f"Inspired by {style.description.lower()} for a unique look. "
        f"Great gift idea for fans of {niche}. Keywords: {sec}. "
        f"Care: machine wash cold, inside-out; tumble dry low."
    )
    return clamp_text(text, 1000)


def make_backend_keywords(primary: List[str], secondary: List[str]) -> str:
    # Space-separated; lowercase; no commas; avoid repeats; aim near byte limit
    terms = list(dict.fromkeys([*(p.lower() for p in primary), *(s.lower() for s in secondary)]))
    # prune obviously redundant words
    stop = {"shirt", "t-shirt", "tee", "for", "and", "the", "a", "gift"}
    terms = [t for t in terms if t not in stop]

    # pack until approx BACKEND_BYTES_TARGET bytes (utf-8)
    packed_tokens: List[str] = []
    total = 0
    for t in terms:
        b = t.encode('utf-8')
        sep = 1 if packed_tokens else 0
        if total + len(b) > BACKEND_BYTES_TARGET:
            break
        total += len(b) + sep
        packed_tokens.append(t)
    return ' '.join(packed_tokens)


def generate_concepts(niche: str, styles_db: List[ArtStyle]) -> List[Concept]:
    primary, secondary = expand_keywords(niche)
    styles = choose_styles(styles_db, niche, k=4)

    concepts: List[Concept] = []
    for s in styles:
        prompt = build_image_prompt(niche, s)
        title = make_title(niche, primary, s)
        desc = make_description(niche, primary, secondary, s)
        backend = make_backend_keywords(primary, secondary)
        concepts.append(Concept(
            niche=niche,
            style=s.style,
            image_prompt=prompt,
            primary_keywords=primary,
            secondary_keywords=secondary,
            title=title,
            description=desc,
            backend_keywords=backend,
        ))
    return concepts


def generate_images_a1111(sd_url: str, concepts: List[Concept], niche: str, steps: int = 28, cfg: float = 6.5, width: int = 768, height: int = 768) -> None:
    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    for idx, c in enumerate(concepts, start=1):
        try:
            payload = {
                "prompt": c.image_prompt,
                # Allow text so quoted phrases can render; still avoid low quality artifacts
                "negative_prompt": "low-res, blurry, jpeg artifacts, watermark, logo, copyright, trademark, misspelling, typos",
                "steps": steps,
                "cfg_scale": cfg,
                "width": width,
                "height": height,
            }
            r = requests.post(f"{sd_url}/sdapi/v1/txt2img", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            if 'images' in data and data['images']:
                b64 = data['images'][0]
                if ',' in b64:
                    b64 = b64.split(',', 1)[-1]
                img_bytes = base64.b64decode(b64)
                out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
                with open(out_path, 'wb') as f:
                    f.write(img_bytes)
                c.image_path = out_path
        except Exception as e:
            print(f"[warn] image generation failed for #{idx}: {e}")


def generate_images_openai(concepts: List[Concept], niche: str, model: str = "gpt-image-1", size: str = "1024x1024") -> None:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not available; install openai and set OPENAI_API_KEY")
    # Support multiple keys: CLI (--openai-keys) not available here, so read from env OPENAI_API_KEYS (comma-separated) and fallback to OPENAI_API_KEY
    keys_raw = _os.environ.get("OPENAI_API_KEYS") or ""
    keys: List[str] = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        single = _os.environ.get("OPENAI_API_KEY")
        if not single:
            raise RuntimeError("Set OPENAI_API_KEY or OPENAI_API_KEYS in environment")
        keys = [single]
    # Map alias 'latest' to a concrete model name via env or default
    if (model or "").lower() == "latest":
        model = _os.environ.get("OPENAI_IMAGE_MODEL_LATEST", "gpt-image-1")

    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    for idx, c in enumerate(concepts, start=1):
        last_err: Exception | None = None
        for api_key in keys:
            try:
                client = OpenAI(api_key=api_key)
                # OpenAI Images API: create image from prompt
                result = client.images.generate(model=model, prompt=c.image_prompt, size=size)
                # API returns base64 in data[0].b64_json
                b64 = result.data[0].b64_json
                img_bytes = base64.b64decode(b64)
                out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
                with open(out_path, 'wb') as f:
                    f.write(img_bytes)
                c.image_path = out_path
                last_err = None
                break
            except Exception as e:
                last_err = e
                # Try next key if available (useful for billing limits or per-key rate limits)
                continue
        if last_err is not None and not c.image_path:
            print(f"[warn] openai image generation failed for #{idx} across all keys: {last_err}")


def _find_b64_values(obj) -> List[str]:
    vals: List[str] = []
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'b64_json' and isinstance(v, str):
                    vals.append(v)
                else:
                    vals.extend(_find_b64_values(v))
        elif isinstance(obj, list):
            for v in obj:
                vals.extend(_find_b64_values(v))
    except Exception:
        pass
    return vals


def generate_images_openai_streaming(concepts: List[Concept], niche: str, model: str = "gpt-image-1", size: str = "1024x1024") -> None:
    """Generate images using OpenAI Images Streaming API via Python SDK.
    Requires openai>=1.2x with images.with_streaming_response.
    It scans streamed events for any 'b64_json' payloads and writes the first image found per concept.
    Falls back across multiple keys via OPENAI_API_KEYS if provided.
    """
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not available; install openai and set OPENAI_API_KEY")
    keys_raw = _os.environ.get("OPENAI_API_KEYS") or ""
    keys: List[str] = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        single = _os.environ.get("OPENAI_API_KEY")
        if not single:
            raise RuntimeError("Set OPENAI_API_KEY or OPENAI_API_KEYS in environment")
        keys = [single]
    if (model or "").lower() == "latest":
        model = _os.environ.get("OPENAI_IMAGE_MODEL_LATEST", "gpt-image-1")

    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    from openai import OpenAI

    for idx, c in enumerate(concepts, start=1):
        last_err: Exception | None = None
        for api_key in keys:
            try:
                client = OpenAI(api_key=api_key)
                # Use streaming response wrapper; request b64_json format to simplify extraction
                with client.images.with_streaming_response.generate(
                    model=model,
                    prompt=c.image_prompt,
                    size=size,
                    response_format="b64_json",
                ) as response:
                    b64_collected: List[str] = []
                    try:
                        # Iterate SDK stream events (SSE). Each event is typically a dict-like chunk.
                        stream = response.parse()
                        for event in stream:
                            # Attempt to discover base64 payloads within this event structure
                            b64s = []
                            if isinstance(event, dict):
                                # Events may have shape {"data": {...}, "event": "..."}
                                b64s.extend(_find_b64_values(event))
                                data = event.get("data")
                                if data is not None:
                                    b64s.extend(_find_b64_values(data))
                            else:
                                # Fallback: try attribute access or direct value
                                try:
                                    data = getattr(event, "data", None)
                                    if data is not None:
                                        b64s.extend(_find_b64_values(data))
                                except Exception:
                                    pass
                            if b64s:
                                b64_collected.extend(b64s)
                    except Exception as stream_err:
                        last_err = stream_err
                        continue

                # If any base64 found, write first; if multiple parts, attempt to join
                if b64_collected:
                    b64_joined = "".join(b64_collected)
                    try:
                        img_bytes = base64.b64decode(b64_joined)
                    except Exception:
                        # If concatenation fails, try first element only
                        img_bytes = base64.b64decode(b64_collected[0])
                    out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
                    with open(out_path, 'wb') as f:
                        f.write(img_bytes)
                    c.image_path = out_path
                    last_err = None
                    break
                else:
                    last_err = RuntimeError("No image data received in stream events")
                    continue
            except Exception as e:
                last_err = e
                continue
        if last_err is not None and not c.image_path:
            print(f"[warn] openai streaming failed for #{idx} across all keys: {last_err}")


def _resolve_gemini_model(name: str) -> str:
    """Map friendly/legacy names to supported Gemini API models for image generation."""
    if not name:
        return "gemini-1.5-flash"
    n = name.strip().lower()
    # Map imagen aliases to 1.5-flash which supports image_generation tool via public API
    if n.startswith("imagen") or n in {"imagegeneration", "image-generation", "image_gen"}:
        return "gemini-1.5-flash"
    return name


def generate_images_gemini(concepts: List[Concept], niche: str, model: str = "imagen-3.0", aspect: str = "1:1") -> None:
    """Generate images with Google Generative Language API using Imagen 3 via generateContent.
    Requires GEMINI_API_KEY.
    """
    api_key = _os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment")

    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    model = _resolve_gemini_model(model)
    base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    for idx, c in enumerate(concepts, start=1):
        try:
            payload = {
                "contents": [{"role": "user", "parts": [{"text": c.image_prompt}]}],
                # Hint: embed aspect ratio in the text prompt; Imagen uses square by default
                "generationConfig": {"response_mime_type": "image/png"}
            }
            r = requests.post(base_url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            # Expected: candidates[0].content.parts[0].inlineData.data (base64)
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError("Gemini returned no candidates")
            parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
            if not parts or not isinstance(parts, list):
                raise RuntimeError("Gemini candidate missing content parts")
            inline = parts[0].get("inlineData") or {}
            b64 = inline.get("data")
            if not b64:
                raise RuntimeError("Gemini candidate missing image data")
            img_bytes = base64.b64decode(b64)
            out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
            with open(out_path, 'wb') as f:
                f.write(img_bytes)
            c.image_path = out_path
        except Exception as e:
            print(f"[warn] gemini image generation failed for #{idx}: {e}")
    


# --- Module-scope helpers for provider order (OpenAI support) ---
def _ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _generate_image_openai_single(prompt: str, out_path: str, model: str, size: str, api_key: str):
    if not _OPENAI_AVAILABLE:
        raise RuntimeError('openai package not available')
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    if (model or "").lower() == "latest":
        model = _os.environ.get("OPENAI_IMAGE_MODEL_LATEST", "gpt-image-1")
    _ensure_parent(out_path)
    resp = client.images.generate(model=model, prompt=prompt, size=size)
    b64 = resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    with open(out_path, 'wb') as f:
        f.write(img_bytes)
    return out_path


def generate_images_failover(concepts: List[Concept], niche: str, args, provider_order: List[str]) -> None:
    """Minimal failover supporting OpenAI and a1111 to satisfy --provider-order usage."""
    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    OPENAI_KEY = _os.environ.get('OPENAI_API_KEY')
    GEMINI_KEY = _os.environ.get('GEMINI_API_KEY')

    for idx, c in enumerate(concepts, start=1):
        prompt = c.image_prompt
        out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
        success = False
        last_err = None

        for provider in provider_order:
            try:
                if provider == 'openai' and OPENAI_KEY:
                    p = _generate_image_openai_single(prompt, out_path, getattr(args, 'openai_model', 'gpt-image-1'), getattr(args, 'openai_size', '1024x1024'), OPENAI_KEY)
                    c.image_path = p
                    success = True
                    break
                elif provider == 'a1111':
                    payload = {
                        "prompt": prompt,
                        "negative_prompt": "low-res, blurry, jpeg artifacts, watermark, logo, copyright, trademark, misspelling, typos",
                        "steps": getattr(args, 'steps', 28),
                        "cfg_scale": getattr(args, 'cfg', 6.5),
                        "width": getattr(args, 'width', 768),
                        "height": getattr(args, 'height', 768),
                    }
                    r = requests.post(f"{args.sd_url.rstrip('/')}/sdapi/v1/txt2img", json=payload, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    images = data.get('images') or []
                    if not images:
                        raise RuntimeError('A1111 returned no images')
                    b64s = images[0]
                    if ',' in b64s:
                        b64s = b64s.split(',', 1)[-1]
                    img_bytes = base64.b64decode(b64s)
                    with open(out_path, 'wb') as f:
                        f.write(img_bytes)
                    c.image_path = out_path
                    success = True
                    break
                elif provider == 'gemini' and GEMINI_KEY:
                    try:
                        gm = _resolve_gemini_model(getattr(args, 'gemini_model', 'imagen-3.0'))
                        base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gm}:generateContent?key={GEMINI_KEY}"
                        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"response_mime_type": "image/png"}}
                        r = requests.post(base_url, json=payload, timeout=120)
                        r.raise_for_status()
                        data = r.json()
                        candidates = data.get("candidates") or []
                        if not candidates:
                            raise RuntimeError("Gemini returned no candidates")
                        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
                        if not parts:
                            raise RuntimeError("Gemini candidate missing content parts")
                        b64 = (parts[0].get("inlineData") or {}).get("data")
                        if not b64:
                            raise RuntimeError("Gemini candidate missing image data")
                        img_bytes = base64.b64decode(b64)
                        with open(out_path, 'wb') as f:
                            f.write(img_bytes)
                        c.image_path = out_path
                        success = True
                        break
                    except Exception as e:
                        last_err = e
                        continue
                else:
                    continue
            except Exception as e:
                last_err = e
                continue

        if not success and last_err:
            print(f"[warn] all providers failed for image #{idx}: {last_err}")


def generate_images_replicate(concepts: List[Concept], niche: str, model: str = "ideogram-ai/ideogram-v3-turbo", aspect: str = "1:1", token: str = "") -> None:
    """Generate images via Replicate. Defaults to Ideogram v3 Turbo model.
    - Reads API token from REPLICATE_API_TOKEN unless provided explicitly via token.
    - Saves the first returned image per concept.
    """
    if not _REPLICATE_AVAILABLE:
        raise RuntimeError("replicate package not available; install replicate and set REPLICATE_API_TOKEN")
    api_token = token or _os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise RuntimeError("REPLICATE_API_TOKEN is not set; pass --replicate-token or export env var")

    # Replicate's Python client reads the token from env var
    _os.environ["REPLICATE_API_TOKEN"] = api_token

    slug = slugify(niche)
    img_dir = os.path.join('output', slug)
    os.makedirs(img_dir, exist_ok=True)

    for idx, c in enumerate(concepts, start=1):
        # Respect free-tier rate limits: ~6/min with burst=1
        if idx > 1:
            time.sleep(10)
        attempts = 0
        while attempts < 3 and not c.image_path:
            attempts += 1
            try:
                output = _replicate.run(model, input={
                    "prompt": c.image_prompt,
                    "aspect_ratio": aspect,
                })
                urls: List[str] = []
                if isinstance(output, list):
                    urls = [u for u in output if isinstance(u, str)]
                elif isinstance(output, dict) and 'output' in output:
                    out = output['output']
                    if isinstance(out, list):
                        urls = [u for u in out if isinstance(u, str)]
                if urls:
                    img_url = urls[0]
                    r = requests.get(img_url, timeout=120)
                    r.raise_for_status()
                    out_path = os.path.join(img_dir, f"img_{idx:02d}.png")
                    with open(out_path, 'wb') as f:
                        f.write(r.content)
                    c.image_path = out_path
                else:
                    if attempts < 3:
                        print(f"[warn] replicate output empty for #{idx}, retrying...")
                        time.sleep(4)
                    else:
                        print(f"[warn] replicate output empty for #{idx} after retries")
            except Exception as e:
                msg = str(e)
                if "429" in msg or "throttled" in msg.lower():
                    wait_s = 6 if attempts == 1 else 12
                    print(f"[warn] replicate rate limited for #{idx}, waiting {wait_s}s and retrying...")
                    time.sleep(wait_s)
                    continue
                else:
                    print(f"[warn] replicate image generation failed for #{idx}: {e}")


        # end replicate helper block


def export_outputs(niche: str, concepts: List[Concept], art_styles: List[ArtStyle]):
    os.makedirs('output', exist_ok=True)
    slug = slugify(niche)

    # Excel for concepts
    rows: List[Dict[str, str]] = []
    for c in concepts:
        rows.append({
            'niche': c.niche,
            'style': c.style,
            'title': c.title,
            'description': c.description,
            'backend_keywords': c.backend_keywords,
            'primary_keywords': ', '.join(c.primary_keywords),
            'secondary_keywords': ', '.join(c.secondary_keywords),
            'image_prompt': c.image_prompt,
            'image_path': c.image_path,
        })
    df = pd.DataFrame(rows)
    out_xlsx = os.path.join('output', f'{slug}_listings.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='listings', index=False)
        # Also include art styles on another sheet
        styles_rows = [{'style': s.style, 'description': s.description, 'example_tags': s.example_tags} for s in art_styles]
        df_styles = pd.DataFrame(styles_rows)
        df_styles.to_excel(writer, sheet_name='art_styles', index=False)

    # CSV outputs
    out_csv = os.path.join('output', f'{slug}_listings.csv')
    df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_xlsx}\nWrote: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="POD listing + prompt generator")
    parser.add_argument("niche", type=str, help="Niche description, e.g., 'funny cat dad'")
    parser.add_argument("--images", action="store_true", help="Generate images as well as text outputs")
    parser.add_argument("--provider", type=str, default="a1111", choices=["a1111","openai","replicate","gemini"], help="Image generation provider")
    parser.add_argument("--sd-url", type=str, default="http://127.0.0.1:7860", help="Stable Diffusion WebUI base URL (provider=a1111)")
    parser.add_argument("--steps", type=int, default=28, help="Sampling steps for image generation")
    parser.add_argument("--cfg", type=float, default=6.5, help="CFG scale")
    parser.add_argument("--width", type=int, default=768, help="Image width")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--openai-model", type=str, default="gpt-image-1", help="OpenAI image model name (provider=openai)")
    parser.add_argument("--openai-size", type=str, default="1024x1024", help="OpenAI image size WxH (provider=openai)")
    parser.add_argument("--openai-stream", action="store_true", help="Use OpenAI Images Streaming API (provider=openai)")
    parser.add_argument("--replicate-model", type=str, default="ideogram-ai/ideogram-v3-turbo", help="Replicate model (provider=replicate)")
    parser.add_argument("--replicate-aspect", type=str, default="1:1", help="Replicate aspect ratio, e.g., 1:1, 3:4, 16:9 (provider=replicate)")
    parser.add_argument("--replicate-token", type=str, default="", help="Replicate API token override; else use REPLICATE_API_TOKEN env")
    # Gemini options
    parser.add_argument("--gemini-model", type=str, default="gemini-1.5-flash", help="Gemini image model id (provider=gemini). Imagen aliases like 'imagen-3.0' will auto-map to a supported model.")
    parser.add_argument("--gemini-aspect", type=str, default="1:1", help="Gemini aspect ratio (provider=gemini), e.g., 1:1, 3:4, 16:9")
    # Multi-key rotation and provider failover
    parser.add_argument("--provider-order", type=str, default=None, help="Comma-separated provider order to try, e.g., 'a1111,openai,replicate'")
    parser.add_argument("--openai-keys", type=str, default=None, help="Comma-separated OpenAI API keys for rotation")
    parser.add_argument("--replicate-tokens", type=str, default=None, help="Comma-separated Replicate tokens for rotation")
    # Load .env before parsing args so env-backed defaults can be honored
    _load_env_from_dotenv()
    args = parser.parse_args()

    niche = args.niche.strip()
    styles_path = os.path.join(os.path.dirname(__file__), 'art_styles.csv')
    if not os.path.exists(styles_path):
        print(f"Missing art styles file at {styles_path}")
        sys.exit(2)

    styles = read_art_styles(styles_path)
    concepts = generate_concepts(niche, styles)

    if args.images:
        # If a provider order is given or multiple keys are present, use failover flow; else preserve legacy behavior
        provider_order = None
        if args.provider_order:
            provider_order = [p.strip() for p in args.provider_order.split(',') if p.strip()]
        else:
            # if rotation flags or env lists are present, enable failover starting with selected provider
            has_multi = bool(args.openai_keys or args.replicate_tokens or _os.environ.get('OPENAI_API_KEYS') or _os.environ.get('REPLICATE_API_TOKENS'))
            if has_multi:
                provider_order = [args.provider]

        if provider_order:
            generate_images_failover(concepts, niche, args, provider_order)
        else:
            if args.provider == "a1111":
                generate_images_a1111(args.sd_url, concepts, niche, steps=args.steps, cfg=args.cfg, width=args.width, height=args.height)
            elif args.provider == "openai":
                if args.openai_stream:
                    generate_images_openai_streaming(concepts, niche, model=args.openai_model, size=args.openai_size)
                else:
                    generate_images_openai(concepts, niche, model=args.openai_model, size=args.openai_size)
            elif args.provider == "replicate":
                generate_images_replicate(concepts, niche, model=args.replicate_model, aspect=args.replicate_aspect, token=args.replicate_token)
            elif args.provider == "gemini":
                generate_images_gemini(concepts, niche, model=args.gemini_model, aspect=args.gemini_aspect)

    export_outputs(niche, concepts, styles)


if __name__ == '__main__':
    main()
