// Summary API using AI SDK with Gemini (if GEMINI_API_KEY/GOOGLE_GENERATIVE_AI_API_KEY is set)
// Falls back to a simple extractive summarizer if not configured.

import { type NextRequest, NextResponse } from "next/server"
import { generateText } from "ai"
import { google } from "@ai-sdk/google"

export async function POST(req: NextRequest) {
  try {
    const { text } = (await req.json()) as { text?: string }
    if (!text || !text.trim()) {
      return NextResponse.json({ error: "Missing text" }, { status: 400 })
    }

    console.log("[v0] summarize text length:", text.length)

    const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_GENERATIVE_AI_API_KEY
    console.log("[v0] summarize key present?", Boolean(apiKey))

    if (apiKey) {
      try {
        const { text: out } = await generateText({
          model: google("gemini-1.5-flash", { apiKey }),
          prompt: [
            "Summarize the following news article in 3-5 concise bullet points.",
            "Focus on who/what/when/where/why; include key numbers if present.",
            "Avoid opinions; be neutral and factual.",
            "",
            text,
          ].join("\n"),
          temperature: 0.2,
        })
        return NextResponse.json({ summary: out })
      } catch (providerErr: any) {
        console.log("[v0] summarize provider error", providerErr?.message || providerErr)
        const fallback = extractiveSummary(text, 5) // bullet-style fallback
        return NextResponse.json({ summary: fallback, providerError: String(providerErr?.message || providerErr) })
      }
    }

    const summary = extractiveSummary(text, 5)
    return NextResponse.json({ summary })
  } catch (e: any) {
    console.error("[v0] summarize error", e)
    return NextResponse.json({ error: "Internal error" }, { status: 500 })
  }
}

function extractiveSummary(text: string, maxPoints = 5): string {
  const normalized = text.replace(/\s+/g, " ").trim()
  if (!normalized) return "- No content provided."

  // Sentence segmentation (robust to missing punctuation)
  const sentenceSplit = normalized.split(/(?<=[.!?])\s+|(?<=;)\s+|(?<=:)\s+/).filter(Boolean)
  const sentences = sentenceSplit.length > 0 ? sentenceSplit : [normalized]

  // Stopwords
  const stop = new Set([
    "the",
    "is",
    "in",
    "and",
    "to",
    "of",
    "a",
    "for",
    "on",
    "that",
    "with",
    "as",
    "by",
    "at",
    "from",
    "be",
    "this",
    "it",
    "an",
    "are",
    "was",
    "were",
    "or",
    "but",
    "not",
    "has",
    "have",
    "had",
    "their",
    "they",
    "his",
    "her",
    "its",
    "you",
    "your",
    "our",
    "them",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
  ])

  // Helpers scoped to fallback (no global pollution)
  const tokenize = (s: string) =>
    (s.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => t.length >= 3 && !stop.has(t))
  const topKeywords = (s: string, n = 5) => {
    const tf = new Map<string, number>()
    for (const t of tokenize(s)) tf.set(t, (tf.get(t) || 0) + 1)
    return [...tf.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([k]) => k)
  }
  const extractNumbersDates = (s: string) => {
    const nums = s.match(/\b(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?\b/g) || []
    const dates =
      s.match(
        /\b(?:\d{4}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t\.|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b/gi,
      ) || []
    return Array.from(new Set([...nums, ...dates]))
  }
  const titleFromText = (s: string) => {
    // Use first sentence/paragraph as base, but compress to headline-like
    const base = sentences[0] || s
    const tokens = tokenize(base).slice(0, 12)
    if (tokens.length === 0) return truncate(base, 80)
    return tokens.join(" ").replace(/^\w/, (c) => c.toUpperCase())
  }

  // If the text is short or sentences are too few, create structured, non-echo bullets
  const isShort = normalized.length < 280 || sentences.length <= 2
  if (isShort) {
    const keywords = topKeywords(normalized, 6)
    const facts = extractNumbersDates(normalized).slice(0, 6)
    const topic = titleFromText(normalized)

    const bullets = [
      `- Topic: ${truncate(topic, 90)}`,
      keywords.length ? `- Key terms: ${keywords.slice(0, 5).join(", ")}` : "",
      facts.length ? `- Facts: ${facts.join(", ")}` : "",
      `- Summary: ${truncate(normalized, 160)}`,
    ].filter(Boolean)

    // Guarantee at least 3 bullets and avoid echoing full text
    while (bullets.length < Math.min(3, maxPoints)) {
      bullets.push("- Context: brief update")
    }

    const out = bullets.slice(0, maxPoints).join("\n")
    return out
  }

  // Standard extractive approach for longer texts
  const freq = new Map<string, number>()
  for (const s of sentences) for (const t of tokenize(s)) freq.set(t, (freq.get(t) || 0) + 1)

  const scored = sentences.map((s, idx) => {
    let score = 0
    for (const t of tokenize(s)) score += freq.get(t) || 0
    return { idx, s, score }
  })

  scored.sort((a, b) => b.score - a.score)
  const targetCount = Math.min(maxPoints, Math.max(3, Math.ceil(sentences.length / 4)))
  const chosenIdxs = scored
    .slice(0, targetCount)
    .map((x) => x.idx)
    .sort((a, b) => a - b)

  const bullets = chosenIdxs.map((i) => "- " + truncate(sentences[i], 180))
  const result = bullets.join("\n")

  // Defensive: never return exact input
  const canonical = (s: string) => s.replace(/\s+/g, " ").trim().toLowerCase()
  if (canonical(result) === canonical(normalized)) {
    return ["- Summary:", "- " + truncate(normalized, 200)].join("\n")
  }
  return result
}

function truncate(s: string, max = 180): string {
  if (s.length <= max) return s
  const cut = s.slice(0, max)
  const lastBreak = Math.max(cut.lastIndexOf("."), cut.lastIndexOf(","), cut.lastIndexOf(" "))
  return (lastBreak > 40 ? cut.slice(0, lastBreak) : cut).trim() + "…"
}
