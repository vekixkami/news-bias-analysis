"use client"

import type React from "react"

import { useState } from "react"
import { BiasResult } from "@/components/bias-result"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useToast } from "@/hooks/use-toast"

type BiasAPIResponse = {
  label: string
  probabilities: { label: string; score: number }[]
  modelInfo: { trainedOn: number; labels: string[] }
}

export default function HomePage() {
  const [text, setText] = useState("")
  const [fileName, setFileName] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [bias, setBias] = useState<BiasAPIResponse | null>(null)
  const [summary, setSummary] = useState<string | null>(null)
  const { toast } = useToast()

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setFileName(f.name)
    if (f.type === "text/plain" || f.name.endsWith(".txt")) {
      const content = await f.text()
      setText(content)
    } else {
      toast({
        title: "Unsupported file",
        description: "Please upload a .txt file or paste the article text directly.",
        variant: "destructive",
      })
      e.target.value = ""
      setFileName(null)
    }
  }

  async function analyze() {
    if (!text.trim()) {
      toast({ title: "Add article text", description: "Please paste the article text or upload a .txt file." })
      return
    }
    setLoading(true)
    setBias(null)
    setSummary(null)
    try {
      // Kick off both requests in parallel
      const biasReq = fetch("/api/bias", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })
      const sumReq = fetch("/api/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })

      const [biasRes, sumRes] = await Promise.all([biasReq, sumReq])

      // Bias handling (non-fatal)
      if (biasRes.ok) {
        const biasJson: BiasAPIResponse = await biasRes.json()
        setBias(biasJson)
      } else {
        toast({
          title: "Bias analysis failed",
          description: "We couldn't estimate bias right now. Summary will still be shown if available.",
          variant: "destructive",
        })
      }

      // Summary handling (independent of bias)
      if (sumRes.ok) {
        const { summary } = (await sumRes.json()) as { summary?: string }
        setSummary(summary || "Summary not available.")
      } else {
        setSummary("Summary service not configured or temporarily unavailable.")
      }
    } catch (err: any) {
      toast({
        title: "Something went wrong",
        description: err?.message || "Failed to analyze article.",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-dvh p-4 md:p-8">
      <div className="mx-auto max-w-3xl space-y-6">
        <header className="space-y-2">
          <div className="inline-flex items-center gap-2 text-xs text-primary">
            <span className="inline-block h-2 w-2 rounded-full bg-primary" aria-hidden="true" />
            <span className="font-medium">Bias analyzer</span>
          </div>
          <h1 className="text-pretty text-3xl font-semibold">News Bias & Summary</h1>
          <p className="text-muted-foreground">
            Upload or paste a news article. We’ll estimate its bias intensity (neutral, slightly-biased, highly-biased)
            and generate a concise summary.
          </p>
        </header>

        <Card className="border-t-4 border-primary">
          <CardHeader>
            <CardTitle>Article</CardTitle>
            <CardDescription>Paste the full article text or upload a .txt file.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-2">
              <Label htmlFor="article">Article Text</Label>
              <Textarea
                id="article"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Please paste your article text here..."
                rows={10}
                className="font-serif"
              />
            </div>
            <div className="flex items-center gap-3">
              <div className="grid gap-1">
                <Label htmlFor="file">Or upload .txt</Label>
                <input id="file" type="file" accept=".txt,text/plain" onChange={handleFile} />
                {fileName && <span className="text-xs text-muted-foreground">Selected: {fileName}</span>}
              </div>
              <div className="flex-1" />
              <Button onClick={analyze} disabled={loading} className="ring-1 ring-primary/20 hover:ring-primary/40">
                {loading ? "Analyzing…" : "Analyze"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {bias && (
          /* accent top border on bias card */
          <Card className="border-t-4 border-primary">
            <CardHeader>
              <CardTitle>Bias Estimation</CardTitle>
              <CardDescription>
                Predicted: <span className="font-medium">{bias.label}</span> • Trained on {bias.modelInfo.trainedOn}{" "}
                samples
              </CardDescription>
            </CardHeader>
            <CardContent>
              <BiasResult probabilities={bias.probabilities} />
              <div className="mt-4 text-sm text-muted-foreground">Labels: {bias.modelInfo.labels.join(", ")}</div>
            </CardContent>
          </Card>
        )}

        {summary && (
          /* accent top border on summary card */
          <Card className="border-t-4 border-primary">
            <CardHeader>
              <CardTitle>Summary</CardTitle>
              <CardDescription>Powered by Gemini if configured, otherwise an extractive fallback.</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="whitespace-pre-wrap">{summary}</p>
            </CardContent>
          </Card>
        )}

        <footer className="text-xs text-muted-foreground">
          Note: Bias estimates are approximate and depend on training samples and preprocessing.
        </footer>
      </div>
    </main>
  )
}
