# POD Design + Amazon SEO Generator

This tool takes a niche input (e.g., "funny cat dad") and generates 4 t-shirt design concepts, each in a different art style, together with SEO-optimized listing content for Amazon Merch on Demand:

- Image prompt for the artwork
- Amazon-ready Title (concise, keyword-rich)
- Description (benefit-led, secondary keywords, natural language)
- Backend Keywords (space-separated, optimized toward the 249-byte guideline)
- Extra: keyword set used and art style notes

It also produces:
- An `art_styles.csv` reference of common t-shirt art styles with short descriptions and example tags
- An Excel workbook `output/<slug>_listings.xlsx` for the 4 generated designs
Optionally, it can also generate images via:
- Local Stable Diffusion WebUI (Automatic1111) API
- OpenAI Images API (DALL·E / gpt-image models)
 - Replicate (e.g., ideogram-ai/ideogram-v3-turbo)

References baked-in follow Amazon SEO best practices from Amazon Seller resources and reputable blogs (see bottom of this README for citations).

## Quick start

1. Create a Python virtual environment and install deps
2. Run the generator with your niche

## Usage

```bash
# text + SEO only
python3 generator.py "your niche here"

# text + SEO + images (Automatic1111)
python3 generator.py "your niche here" --images --provider a1111 --sd-url http://127.0.0.1:7860

# text + SEO + images (OpenAI)
# Requires env var OPENAI_API_KEY
export OPENAI_API_KEY=sk-...
python3 generator.py "your niche here" --images --provider openai --openai-model gpt-image-1 --openai-size 1024x1024

# text + SEO + images (Replicate + Ideogram)
# Requires env var REPLICATE_API_TOKEN or pass --replicate-token
export REPLICATE_API_TOKEN=r8_...
python3 generator.py "your niche here" --images --provider replicate \
  --replicate-model ideogram-ai/ideogram-v3-turbo \
  --replicate-aspect 1:1
```

Outputs will be written to `output/` as an Excel file and CSV.

## What it does

- Normalizes your niche and expands it into related keyword clusters (short-tail + long-tail)
- Picks 4 distinct, relevant art styles
- Crafts 4 image-generation prompts (style + subject + composition + color + print constraints)
- Writes Amazon-ready SEO fields:
  - Title: primary keyword + core descriptor; <= 200 chars recommendation, aiming ~60-120
  - Description: secondary keywords, features/benefits, care notes; avoids keyword stuffing
  - Backend keywords: space-separated synonyms, stems, misspellings avoided; aims near 249 bytes
- Exports to Excel with clear columns

## Cautions and Policy

- Don’t include prohibited/claim-heavy language (medical claims, IP/copyrighted brands/characters, adult/hate content, etc.)
- Avoid keyword stuffing or repetition in title and description
- Use relevant keywords only; no competitor brand names
- Ensure you own the rights to any artwork you upload

## Sources and guidance
- Amazon Selling Partner Blog: "Amazon SEO: 7 ways to improve your product’s search rankings" (2025)
- Jungle Scout: "Improve Your Product Listings (and Rank Higher) with Amazon SEO" (2024)
- Helium 10: "Steal Our Amazon SEO Strategy (Based on A9 Algorithms)" (context on indexing and hierarchy)
- Printify: "56 T-shirt design ideas" (for art style coverage)

To generate images, ensure you have either:
- Automatic1111: WebUI running with `--api` and set `--provider a1111` (default) with `--sd-url`
- OpenAI: Set `OPENAI_API_KEY` and use `--provider openai` with optional `--openai-model` and `--openai-size` flags
 - Replicate: Set `REPLICATE_API_TOKEN` (or use `--replicate-token`) and `--provider replicate`. Default model is `ideogram-ai/ideogram-v3-turbo`.

Images are written to `output/<slug>/img_<index>.png` and the path is included in the Excel/CSV.

This repository does not fetch live data; it encodes best practices and style taxonomies for a repeatable workflow.
