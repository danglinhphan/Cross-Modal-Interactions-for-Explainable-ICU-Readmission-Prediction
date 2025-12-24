import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { TrendingUp, TrendingDown } from "lucide-react"

interface Feature {
  name: string
  impact: number
  value: string
}

interface QuantitativeExplanationProps {
  features: Feature[]
}

export function QuantitativeExplanation({ features }: QuantitativeExplanationProps) {
  const chartData = features.map((f) => ({
    name: f.name,
    impact: f.impact,
    value: f.value,
  }))

  return (
    <Card className="border-white/10 bg-[#0d1117]/60 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-white">Quantitative Analysis</CardTitle>
        <CardDescription className="text-gray-400">Feature importance from EBM model</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mb-6 h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 120, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
              <XAxis type="number" stroke="#9ca3af" />
              <YAxis type="category" dataKey="name" stroke="#9ca3af" width={110} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1f35",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  color: "#fff",
                }}
                formatter={(value: number, name: string, props: any) => [
                  `${value.toFixed(1)} points`,
                  props.payload.value,
                ]}
              />
              <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.impact > 0 ? "#ef4444" : "#10b981"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-white">Feature Breakdown</h3>
          {features.map((feature, index) => (
            <div key={index} className="flex items-center justify-between rounded-lg bg-white/5 p-3">
              <div className="flex items-center gap-3">
                {feature.impact > 0 ? (
                  <TrendingUp className="size-4 text-red-400" />
                ) : (
                  <TrendingDown className="size-4 text-emerald-400" />
                )}
                <div>
                  <div className="text-sm font-medium text-white">{feature.name}</div>
                  <div className="text-xs text-gray-400">{feature.value}</div>
                </div>
              </div>
              <div className={`text-sm font-semibold ${feature.impact > 0 ? "text-red-400" : "text-emerald-400"}`}>
                {feature.impact > 0 ? "+" : ""}
                {feature.impact.toFixed(1)}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
