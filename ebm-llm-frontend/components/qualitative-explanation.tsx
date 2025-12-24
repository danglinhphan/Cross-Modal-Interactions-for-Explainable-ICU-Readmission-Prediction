import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Sparkles } from "lucide-react"

interface QualitativeExplanationProps {
  explanation: string
}

export function QualitativeExplanation({ explanation }: QualitativeExplanationProps) {
  return (
    <Card className="border-white/10 bg-[#0d1117]/60 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-white">
          <Sparkles className="size-5 text-[#7C3AED]" />
          Clinical Interpretation
        </CardTitle>
        <CardDescription className="text-gray-400">AI-generated explanation from LLM analysis</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="rounded-lg bg-gradient-to-br from-[#4F46E5]/10 to-[#7C3AED]/10 p-6 leading-relaxed text-gray-300">
            {explanation.split("\n\n").map((paragraph, index) => (
              <p key={index} className="mb-4 last:mb-0">
                {paragraph}
              </p>
            ))}
          </div>

          <div className="flex items-start gap-3 rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-4">
            <div className="mt-0.5 text-yellow-500">
              <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <div className="text-sm text-gray-300">
              <p className="font-medium text-yellow-400">Clinical Note</p>
              <p className="mt-1 text-gray-400">
                This AI-generated interpretation should be reviewed by qualified medical professionals before making
                clinical decisions.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
