import { NextResponse } from "next/server"

export const dynamic = "force-dynamic"

type Label = "neutral" | "slightly-biased" | "highly-biased"

type DatasetRow = { text: string; label: Label }

type BiasModel = {
  labels: Label[]
  vocab: Set<string>
  docCount: Record<Label, number>
  tokenCounts: Record<Label, Map<string, number>>
  totalTokens: Record<Label, number>
  trainedOn: number
}

let cachedModel: BiasModel | null = null
let buildingPromise: Promise<BiasModel> | null = null

function normalizeLabel(raw: string): Label | null {
  const r = raw.trim().toLowerCase()
  if (r.includes("neutral")) return "neutral"
  if (r.includes("slight")) return "slightly-biased"
  if (r.includes("high") || r.includes("extreme") || r.includes("strong")) return "highly-biased"
  return null
}

function tokenize(text: string): string[] {
  // Lowercase, keep alpha tokens, drop short tokens and common stopwords
  const stop = new Set([
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "has",
    "had",
    "are",
    "was",
    "were",
    "but",
    "not",
    "you",
    "your",
    "our",
    "their",
    "they",
    "his",
    "her",
    "its",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "will",
    "would",
    "could",
    "should",
    "said",
    "says",
    "say",
    "into",
    "over",
    "under",
    "than",
    "then",
    "also",
    "more",
    "most",
    "some",
    "any",
    "all",
    "because",
    "been",
    "after",
    "before",
    "about",
    "against",
    "between",
    "during",
    "without",
    "within",
    "while",
    "into",
    "onto",
    "off",
    "per",
    "via",
    "as",
    "by",
    "at",
    "on",
    "in",
    "to",
    "of",
    "a",
    "an",
    "it",
  ])
  const tokens = text.toLowerCase().match(/[a-z]+/g) || []
  return tokens.filter((t) => t.length >= 3 && !stop.has(t))
}

function trainNaiveBayes(rows: DatasetRow[]): BiasModel {
  const labels: Label[] = ["neutral", "slightly-biased", "highly-biased"]
  const vocab = new Set<string>()
  const docCount = {
    neutral: 0,
    "slightly-biased": 0,
    "highly-biased": 0,
  } as Record<Label, number>
  const tokenCounts: Record<Label, Map<string, number>> = {
    neutral: new Map(),
    "slightly-biased": new Map(),
    "highly-biased": new Map(),
  }
  const totalTokens: Record<Label, number> = {
    neutral: 0,
    "slightly-biased": 0,
    "highly-biased": 0,
  }

  for (const { text, label } of rows) {
    const toks = Array.from(new Set(tokenize(text))) // binary NB: presence, not frequency
    docCount[label]++
    for (const t of toks) {
      vocab.add(t)
      tokenCounts[label].set(t, (tokenCounts[label].get(t) || 0) + 1)
      totalTokens[label]++
    }
  }

  console.log("[v0] bias training: rows", rows.length, "vocab", vocab.size, "docCount", docCount)
  return { labels, vocab, docCount, tokenCounts, totalTokens, trainedOn: rows.length }
}

function softmaxLogScores(scores: Record<Label, number>): Record<Label, number> {
  const vals = Object.values(scores)
  const max = Math.max(...vals)
  const exps = Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, Math.exp(v - max)])) as Record<
    Label,
    number
  >
  const sum = Object.values(exps).reduce((a, b) => a + b, 0)
  const probs = Object.fromEntries(Object.entries(exps).map(([k, v]) => [k, v / (sum || 1)])) as Record<Label, number>
  return probs
}

function predict(model: BiasModel, text: string): { label: Label; probs: Record<Label, number> } {
  const { labels, vocab, docCount, tokenCounts, totalTokens } = model
  const N = labels.reduce((acc, l) => acc + docCount[l], 0)
  const V = Math.max(1, vocab.size)
  const toks = Array.from(new Set(tokenize(text)))

  const logScores = {} as Record<Label, number>
  for (const l of labels) {
    // Laplace-smoothed prior
    const prior = Math.log((docCount[l] + 1) / (N + labels.length))
    let ll = prior
    for (const t of toks) {
      const count = tokenCounts[l].get(t) || 0
      const prob = (count + 1) / (totalTokens[l] + V)
      ll += Math.log(prob)
    }
    logScores[l] = ll
  }

  const probs = softmaxLogScores(logScores)
  let best: Label = labels[0]
  let bestVal = Number.NEGATIVE_INFINITY
  for (const l of labels) {
    if (logScores[l] > bestVal) {
      bestVal = logScores[l]
      best = l
    }
  }
  return { label: best, probs }
}

// --- CSV parsing and gated fetch helpers ---

function parseCSV(text: string): string[][] {
  const rows: string[][] = []
  let i = 0,
    field = "",
    row: string[] = [],
    inQuotes = false
  while (i < text.length) {
    const c = text[i]
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i += 2
          continue
        }
        inQuotes = false
        i++
        continue
      }
      field += c
      i++
      continue
    } else {
      if (c === '"') {
        inQuotes = true
        i++
        continue
      }
      if (c === ",") {
        row.push(field)
        field = ""
        i++
        continue
      }
      if (c === "\n") {
        row.push(field)
        rows.push(row)
        row = []
        field = ""
        i++
        continue
      }
      if (c === "\r") {
        i++
        continue
      }
      field += c
      i++
      continue
    }
  }
  row.push(field)
  rows.push(row)
  if (rows.length && rows[rows.length - 1].length === 1 && rows[rows.length - 1][0] === "") rows.pop()
  return rows
}

async function fetchRawCsvRows(limit = 2000): Promise<DatasetRow[]> {
  const token = process.env.HUGGINGFACE_TOKEN || process.env.HF_TOKEN
  const headers = token ? { Authorization: `Bearer ${token}` } : undefined
  // Try common raw CSV names in the repo
  const urls = [
    "https://huggingface.co/datasets/newsmediabias/news-bias-full-data/resolve/main/train.csv",
    "https://huggingface.co/datasets/newsmediabias/news-bias-full-data/resolve/main/data.csv",
    "https://huggingface.co/datasets/newsmediabias/news-bias-full-data/resolve/main/news_bias_full.csv",
    "https://huggingface.co/datasets/newsmediabias/news-bias-full-data/resolve/main/news-bias-full-data.csv",
  ]
  for (const url of urls) {
    try {
      const r = await fetch(url, { cache: "no-store", headers })
      console.log("[v0] bias raw fetch:", url, "->", r.status, "token?", !!token)
      if (!r.ok) continue
      const csv = await r.text()
      const rows = parseCSV(csv)
      if (!rows.length) continue
      const header = rows[0].map((h) => h.trim().toLowerCase())
      const textIdx = header.findIndex((h) => ["text", "article", "content", "body"].includes(h))
      const labelIdx = header.findIndex((h) => ["label", "bias", "leaning", "dimension"].includes(h))
      if (textIdx === -1 || labelIdx === -1) continue

      const wanted = new Set<Label>(["neutral", "slightly-biased", "highly-biased"])
      const out: DatasetRow[] = []
      for (let i = 1; i < rows.length && out.length < limit; i++) {
        const r = rows[i]
        const text = (r[textIdx] || "").toString()
        const raw = (r[labelIdx] || "").toString().toLowerCase()
        const norm = normalizeLabel(raw)
        if (!text || !norm || !wanted.has(norm)) continue
        out.push({ text, label: norm })
      }
      if (out.length) return out
    } catch (e: any) {
      console.log("[v0] bias raw fetch error:", e?.message || e)
    }
  }
  return []
}

function balancedSeed(): DatasetRow[] {
  const neutral = [
    "The report outlines the timeline of events and quotes multiple sources without editorializing.",
    "Officials provided details about the incident; no conclusions were drawn pending investigation.",
    "Data from the study is presented with methodology and limitations clearly described.",
    "The article summarizes statements from both parties without implying motives.",
    "Key facts were verified against public records and official documents.",
    "The briefing covered the policy proposal with quotes from proponents and opponents.",
    "Market results are reported with historical context and analyst perspectives.",
    "International reactions are listed with minimal interpretation.",
    "The court filing is summarized with references to the original documents.",
    "The piece outlines procedural steps taken by the committee this week.",
  ]
  const slight = [
    "Experts say the plan could raise concerns, though supporters downplay the risk.",
    "Critics argue the measure may go too far, while backers call it a necessary step.",
    "The article frames the issue as contentious, highlighting possible drawbacks.",
    "Some observers contend the language in the bill is vague and open to abuse.",
    "Opponents questioned the timing, suggesting political motives could be at play.",
    "Supporters maintain the change is overdue despite logistical challenges.",
    "Analysts warn there might be unintended consequences in certain regions.",
    "Commentators note the messaging strategy favors one side’s narrative.",
    "Some reports emphasize the dispute more than the underlying data.",
    "Coverage points out inconsistencies without fully exploring alternative views.",
  ]
  const high = [
    "The piece denounces opponents in strong terms, suggesting malicious intent.",
    "Language repeatedly labels one side as dishonest without presenting evidence.",
    "The article uses charged descriptions to portray the policy as catastrophic.",
    "Opposing views are dismissed as propaganda rather than addressed on merits.",
    "Hyperbolic claims are made with little sourcing, implying a foregone conclusion.",
    "The narrative implies conspiratorial motives throughout the coverage.",
    "Selective quoting and loaded adjectives frame the debate as a moral battle.",
    "The report ridicules certain groups while ignoring counter-evidence.",
    "It amplifies unverified allegations and treats speculation as fact.",
    "One-sided framing presents dissenting experts as bad-faith actors.",
  ]
  const rows: DatasetRow[] = []
  for (const t of neutral) rows.push({ text: t, label: "neutral" })
  for (const t of slight) rows.push({ text: t, label: "slightly-biased" })
  for (const t of high) rows.push({ text: t, label: "highly-biased" })
  return rows.concat(rows) // duplicate to increase counts
}

async function buildModel(): Promise<BiasModel> {
  console.log("[v0] bias training: start")
  let rows: DatasetRow[] = []
  try {
    rows = await fetchRawCsvRows(2000)
  } catch {}
  if (!rows.length) {
    console.log(
      "[v0] bias training: using balanced seed fallback (dataset gated; set HF_TOKEN/HUGGINGFACE_TOKEN after accepting terms)",
    )
    rows = balancedSeed()
  }
  const model = trainNaiveBayes(rows)
  console.log("[v0] bias training: done. rows:", rows.length, "labels:", model.labels, "vocab:", model.vocab.size)
  return model
}

async function getModel(): Promise<BiasModel> {
  if (cachedModel) return cachedModel
  if (buildingPromise) return buildingPromise
  buildingPromise = buildModel()
    .then((m) => {
      cachedModel = m
      buildingPromise = null
      return m
    })
    .catch((e) => {
      buildingPromise = null
      throw e
    })
  return buildingPromise
}

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}))
    const text = (body?.text as string) || ""
    if (!text || !text.trim()) {
      return NextResponse.json({ error: "Missing text" }, { status: 400 })
    }

    const model = await getModel()
    const { label, probs } = predict(model, text)

    const probabilities = model.labels.map((l) => ({ label: l, score: probs[l] })).sort((a, b) => b.score - a.score)

    return NextResponse.json({
      label,
      probabilities,
      modelInfo: { trainedOn: model.trainedOn, labels: model.labels },
    })
  } catch (e: any) {
    console.error("[v0] bias error:", e?.message || e)
    return NextResponse.json({ error: "Internal error" }, { status: 500 })
  }
}
