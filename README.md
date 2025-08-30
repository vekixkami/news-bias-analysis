# News Bias Analysis – Technical Documentation

This app lets users paste or upload a news article, then:
- Estimate bias intensity (neutral, slightly-biased, highly-biased) using a lightweight Naive Bayes classifier
- Generate a concise summary via Gemini (if configured) or an extractive fallback

It is built with the Next.js App Router and Shadcn UI components.

- UI: app/page.tsx
- Bias API: app/api/bias/route.ts
- Summary API: app/api/summarize/route.ts

Note: The Hugging Face dataset newsmediabias/news-bias-full-data is gated (terms required). Without a token, the app trains on a balanced local seed so results vary, but they won’t match the full dataset’s performance.

----------------------------------------------------------------

## Quick Start

1) Set environment variables in Project Settings:
   - HF_TOKEN or HUGGINGFACE_TOKEN (optional but recommended)
     - Required to train on the gated dataset (after accepting conditions on Hugging Face).
   - GEMINI_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY (optional, for LLM summaries)

2) Open the app, paste or upload a .txt article, and click Analyze.
   - Bias and Summary run in parallel; each can succeed independently.

3) Check logs if needed
   - The APIs log with a [v0] prefix (e.g., training stats, token presence, fetch status codes).

----------------------------------------------------------------

## Environment Variables

- HF_TOKEN or HUGGINGFACE_TOKEN
  - Used to authenticate against the gated Hugging Face dataset for CSV fetches.
  - You must log in to Hugging Face and accept the dataset’s conditions first.

- GEMINI_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY
  - Enables Gemini via the AI SDK (@ai-sdk/google) for abstractive summarization.
  - Without a key, the app uses a deterministic extractive fallback (bullet points).

Notes
- Environment variables are server-only in this runtime. Client code cannot read them.
- If both HF_TOKEN and HUGGINGFACE_TOKEN are present, either may be used.

----------------------------------------------------------------

## APIs

### POST /api/bias

Request
- Body JSON: { "text": string } (required)

Response (200)
- label: string – predicted top class ("neutral" | "slightly-biased" | "highly-biased")
- probabilities: Array<{ label: string; score: number }>
  - Same labels as above, scores sum to ~1
- modelInfo: { trainedOn: number; labels: string[] }

Example
\`\`\`
POST /api/bias
Content-Type: application/json

{ "text": "Your article text..." }
\`\`\`

Example response
\`\`\`
{
  "label": "slightly-biased",
  "probabilities": [
    { "label": "slightly-biased", "score": 0.62 },
    { "label": "neutral", "score": 0.25 },
    { "label": "highly-biased", "score": 0.13 }
  ],
  "modelInfo": { "trainedOn": 1200, "labels": ["neutral","slightly-biased","highly-biased"] }
}
\`\`\`

Failure responses
- 400: { "error": "Missing text" }
- 500: { "error": "Internal error" }

Implementation details (app/api/bias/route.ts)
- dynamic = "force-dynamic" to ensure fresh behavior in this runtime
- Training source
  - Preferred: Raw CSV from the Hugging Face dataset (resolve/main + common CSV names)
    - Authorization header set when HF token is present
    - CSV parser implemented locally (parseCSV) to avoid Node-only libs
    - Header detection: finds Text-like column (text/article/content/body) and Label-like column (label/bias/leaning/dimension)
    - Label normalization (normalizeLabel) maps arbitrary values to:
      - "neutral"
      - "slightly-biased"
      - "highly-biased"
  - Fallback: balancedSeed() if remote fetch fails or no rows parsed
    - Seed includes examples across all classes and is duplicated to increase counts
- Model: Lightweight Naive Bayes (binary presence)
  - Tokenization: lowercase, alpha tokens, short-word + stopword filtering
  - Uses binary presence per document for counts (Bernoulli-like)
  - Laplace smoothing on priors and likelihoods
  - Softmax over log-scores to return probabilities
- Caching:
  - Module-level singleton (cachedModel) + in-flight promise guard (buildingPromise)
  - Training occurs on first request; subsequent requests reuse the model until cold start
- Logging:
  - [v0] bias raw fetch: URL, HTTP status, token present?
  - [v0] bias training: rows, vocab size, docCount
  - [v0] bias error: errors during processing

Practical notes
- With no HF token or without accepting terms, the service uses the balanced fallback. Predictions will vary by input, but the model is intentionally simple and not equivalent to the gated dataset’s performance.
- If predictions look constant, check logs for "using balanced seed fallback" and ensure HF_TOKEN/HUGGINGFACE_TOKEN is set and dataset terms are accepted.

---

### POST /api/summarize

Request
- Body JSON: { "text": string } (required)

Response (200)
- { "summary": string }
  - If a Gemini key is present, summary is LLM-generated via the AI SDK.
  - Otherwise, a deterministic extractive bullet list is returned.
- When Gemini errors occur, the API falls back to extractive summary and returns 200, optionally including providerError in the payload.

Failure responses
- 400: { "error": "Missing text" }
- 500: { "error": "Internal error" }

Implementation details (app/api/summarize/route.ts)
- Uses AI SDK’s generateText with @ai-sdk/google when GEMINI_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY is present
  - Model: gemini-1.5-flash
  - Temperature: 0.2
- Extractive fallback:
  - Sentence splitting + term-frequency scoring
  - Selects 3..5 top sentences (in input order) and renders bullet points
  - Truncates long bullets; never blindly echoes the entire input
- Logging:
  - [v0] summarize text length
  - [v0] summarize key present? true/false
  - [v0] summarize provider error on failure

----------------------------------------------------------------

## UI (app/page.tsx)

- Lets the user paste article text or upload a .txt file (validated MIME/extension).
- Clicking Analyze triggers both APIs in parallel:
  - Bias: POST /api/bias
  - Summary: POST /api/summarize
- Each result renders independently so one failure does not block the other.
- Bias output:
  - Predicted label
  - Probability bars via BiasResult (component must accept probabilities: { label, score }[])
  - Model metadata (trainedOn count, label list)
- Summary output:
  - Text response shown as pre-wrapped content
  - Card descriptions clarify the provider behavior (Gemini or fallback)

----------------------------------------------------------------

## Design Notes

- Color palette: light/dark tokens defined in app/globals.css
  - Primary brand color: blue (#2563eb)
  - Neutrals: background/foreground variants
  - No gradients
- Typography: Geist (Sans + Mono) loaded in app/layout.tsx
  - Body uses font-sans
- Layout: mobile-first, simple cards and spacing using Tailwind classes
  - Uses flex/grid patterns, semantic elements, accessible labels

----------------------------------------------------------------

## Caching & Performance

- Bias model is trained once per warm runtime and cached in module scope
- Training cost depends on sample size
  - CSV sampling limit is ~2000 rows by default
- If CSV fetch is gated or fails, a balanced fallback is used to avoid degenerate priors

----------------------------------------------------------------

## Troubleshooting

Bias predictions look the same
- Cause: model trained on fallback only or insufficient variety in tokens
- Actions:
  - Ensure you accepted the dataset terms on Hugging Face
  - Add HF_TOKEN or HUGGINGFACE_TOKEN in environment variables
  - Re-run so training pulls real rows (watch [v0] logs for “bias raw fetch: ... -> 200” and “trainedOn” > fallback size)
  - Verify your input is sufficiently long and content-rich

Summary is just my input
- Cause: short input, or fallback previously allowed echoes
- Actions:
  - Provide GEMINI_API_KEY (or GOOGLE_GENERATIVE_AI_API_KEY) to enable Gemini
  - The extractive fallback selects salient sentences and truncates; very short inputs may produce one short bullet

Dataset fetch errors (401/422)
- Cause: gated dataset or missing/invalid HF token
- Actions: accept terms on Hugging Face and set HF_TOKEN/HUGGINGFACE_TOKEN, then try again

500 errors or nothing renders
- Check server logs for [v0] messages (e.g., "bias error" or "summarize provider error")
- Ensure files aren’t truncated; a partial route.ts will fail to compile

----------------------------------------------------------------

## cURL Examples

Bias
\`\`\`
curl -s -X POST http://localhost:3000/api/bias \
  -H "Content-Type: application/json" \
  -d '{"text":"Your article text here..."}' | jq
\`\`\`

Summary
\`\`\`
curl -s -X POST http://localhost:3000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text":"Your article text here..."}' | jq
\`\`\`

----------------------------------------------------------------

## Roadmap

- Stronger baseline: TF‑IDF + logistic regression
- Better tokenization: n-grams, lemmatization, stopword lists per domain
- Calibration and confidence metrics
- Article URL ingestion + readability pipeline
- Persisted model snapshots and warm-start across deploys
