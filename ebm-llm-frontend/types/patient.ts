export interface PatientData {
  id: string
  Subject_ID?: string
  HADM_ID?: string
  [key: string]: any
}

export interface Feature {
  name: string
  impact: number
  value: string
}

export interface PredictionResult {
  riskScore: number
  confidence: number
  features: Feature[]
  llmExplanation: string
}
