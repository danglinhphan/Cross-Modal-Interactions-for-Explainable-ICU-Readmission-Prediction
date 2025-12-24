import { Card, CardContent } from "@/components/ui/card"
import { AlertTriangle, Clock } from "lucide-react"

interface RiskScoreDisplayProps {
  riskScore: number
  confidence: number
}

export function RiskScoreDisplay({ riskScore, confidence }: RiskScoreDisplayProps) {
  const getRiskLevel = (score: number) => {
    if (score < 30) return { label: "Low Risk", color: "text-emerald-400", bgColor: "bg-emerald-500/10" }
    if (score < 70) return { label: "Moderate Risk", color: "text-yellow-400", bgColor: "bg-yellow-500/10" }
    return { label: "High Risk", color: "text-red-400", bgColor: "bg-red-500/10" }
  }

  const riskLevel = getRiskLevel(riskScore)

  return (
    <Card className="border-white/10 bg-[#0d1117]/60 backdrop-blur-sm">
      <CardContent className="p-8">
        <div className="grid gap-8 md:grid-cols-2">
          {/* Risk Score */}
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="flex items-center gap-2 text-gray-400">
              <Clock className="size-5" />
              <span className="text-sm font-medium">Readmission Risk Score</span>
            </div>
            <div
              className={`flex size-40 items-center justify-center rounded-full ${riskLevel.bgColor} border-2 border-current ${riskLevel.color}`}
            >
              <div className="text-center">
                <div className="text-5xl font-bold">
                  {riskScore.toFixed(1)}
                  <span className="text-2xl">%</span>
                </div>
                <div className="mt-1 text-sm font-medium">{riskLevel.label}</div>
              </div>
            </div>
          </div>

          {/* Confidence Level */}
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="flex items-center gap-2 text-gray-400">
              <AlertTriangle className="size-5" />
              <span className="text-sm font-medium">Model Confidence</span>
            </div>
            <div className="w-full space-y-3">
              <div className="text-center">
                <div className="text-5xl font-bold text-[#4F46E5]">
                  {confidence.toFixed(1)}
                  <span className="text-2xl">%</span>
                </div>
              </div>
              <div className="h-3 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-[#4F46E5] to-[#7C3AED] transition-all duration-500"
                  style={{ width: `${confidence}%` }}
                />
              </div>
              <p className="text-center text-sm text-gray-400">High confidence in prediction accuracy</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
