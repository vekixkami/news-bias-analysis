"use client"
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts"

export function BiasResult({
  probabilities,
}: {
  probabilities: { label: string; score: number }[]
}) {
  const data = probabilities.map((p) => ({ label: p.label, score: Number((p.score * 100).toFixed(2)) }))
  return (
    <div className="w-full">
      <div className="h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ left: 8, right: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" />
            <YAxis unit="%" />
            <Tooltip formatter={(value) => `${value}%`} />
            <Bar dataKey="score" fill="var(--color-primary)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
