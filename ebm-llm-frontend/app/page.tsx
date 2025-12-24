"use client"

import type React from "react"

import { useState } from "react"
import { Upload, AlertCircle, Activity } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { RiskScoreDisplay } from "@/components/risk-score-display"
import { QuantitativeExplanation } from "@/components/quantitative-explanation"
import { QualitativeExplanation } from "@/components/qualitative-explanation"
import { PatientDataTable } from "@/components/patient-data-table"
import type { PatientData, PredictionResult } from "@/types/patient"

export default function HealthcareDashboard() {
  const [patients, setPatients] = useState<PatientData[]>([])
  const [selectedPatientId, setSelectedPatientId] = useState<string>("")
  const [selectedPatient, setSelectedPatient] = useState<PatientData | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [isDataPreviewOpen, setIsDataPreviewOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [nlpHighlights, setNlpHighlights] = useState<{ feature: string, text: string, start: number, end: number }[]>([])
  const [activeNlpFeatures, setActiveNlpFeatures] = useState<string[]>([])

  // Helper to render highlighted clinical notes
  const renderHighlightedNotes = (notes: string, highlights: typeof nlpHighlights) => {
    if (!notes || highlights.length === 0) return notes

    // Sort highlights by position (reverse to process from end)
    const sorted = [...highlights].sort((a, b) => b.start - a.start)
    let result = notes

    // Create highlighted HTML
    for (const h of sorted) {
      const before = result.slice(0, h.start)
      const match = result.slice(h.start, h.end)
      const after = result.slice(h.end)
      result = `${before}<mark class="bg-yellow-500/30 text-yellow-200 px-1 rounded" title="${h.feature}">${match}</mark>${after}`
    }
    return result
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    console.log("📁 File selected:", file.name, "Size:", (file.size / 1024 / 1024).toFixed(2), "MB")

    const text = await file.text()
    const rows = text.split("\n")
    const headers = rows[0].split(",").map(h => h.trim().replace(/"/g, ""))

    console.log("📋 Headers found:", headers.length)

    // Limit to first 100 patients for performance
    const MAX_PATIENTS = 100
    const dataRows = rows.slice(1).filter((row) => row.trim())
    const limitedRows = dataRows.slice(0, MAX_PATIENTS)

    // Proper CSV parser that handles quoted fields with commas
    const parseCSVLine = (line: string): string[] => {
      const result: string[] = []
      let current = ''
      let inQuotes = false

      for (let i = 0; i < line.length; i++) {
        const char = line[i]
        if (char === '"') {
          inQuotes = !inQuotes
        } else if (char === ',' && !inQuotes) {
          result.push(current.trim().replace(/^"|"$/g, ''))
          current = ''
        } else {
          current += char
        }
      }
      result.push(current.trim().replace(/^"|"$/g, ''))
      return result
    }

    const parsedPatients: PatientData[] = limitedRows.map((row, index) => {
      // Use proper CSV parser that handles quoted fields with commas
      const values = parseCSVLine(row)
      const patientData: any = { id: `patient-${index}` }
      headers.forEach((header, i) => {
        const value = values[i] || ""
        // Keep clinical_notes as string, try numeric for others
        if (header === 'clinical_notes' || header === 'CLEAN_TEXT') {
          patientData[header] = value
        } else {
          const numValue = parseFloat(value)
          patientData[header] = isNaN(numValue) ? value : numValue
        }
      })
      return patientData as PatientData
    })

    console.log("✅ Parsed:", parsedPatients.length, "of", dataRows.length, "patients")

    // Log clinical_notes length for debugging
    if (parsedPatients.length > 0 && parsedPatients[0].clinical_notes) {
      console.log("📝 Clinical notes sample length:", parsedPatients[0].clinical_notes.length, "chars")
    }

    setPatients(parsedPatients)
    alert(`✅ Loaded ${parsedPatients.length} of ${dataRows.length} patients (max ${MAX_PATIENTS})`)
  }

  const handlePatientSelect = async (patientId: string) => {
    setSelectedPatientId(patientId)
    const patient = patients.find((p) => p.id === patientId)
    setSelectedPatient(patient || null)
    setPrediction(null)  // Clear previous prediction
    setNlpHighlights([])  // Clear previous highlights
    setActiveNlpFeatures([])  // Clear previous active features
    setIsLoading(true)

    // Call real EBM + LLM API
    if (patient) {
      try {
        // Prepare patient data (remove id field)
        const { id, ...patientFeatures } = patient

        console.log("🔍 Patient features:", patientFeatures)

        // Extract clinical_notes if present
        const clinicalNotes = patientFeatures.clinical_notes || patientFeatures.DISCHARGE_SUMMARY_TEXT || ""
        delete patientFeatures.clinical_notes
        delete patientFeatures.DISCHARGE_SUMMARY_TEXT

        // Call NLP extraction for highlights
        if (clinicalNotes) {
          try {
            const nlpResponse = await fetch("http://localhost:8000/extract-nlp", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ clinical_notes: clinicalNotes })
            })
            if (nlpResponse.ok) {
              const nlpResult = await nlpResponse.json()
              setNlpHighlights(nlpResult.highlights)
              setActiveNlpFeatures(Object.entries(nlpResult.features)
                .filter(([_, v]) => v === 1)
                .map(([k, _]) => k))
              console.log("🔍 NLP features:", nlpResult.active_count, "active")
            }
          } catch (e) {
            console.warn("NLP extraction failed:", e)
          }
        }

        console.log("📤 Sending to API:", { patient_data: patientFeatures, clinical_notes: clinicalNotes?.substring(0, 100) })

        const response = await fetch("http://localhost:8000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            patient_data: patientFeatures,
            clinical_notes: clinicalNotes,
            generate_llm_explanation: true
          })
        })

        console.log("📥 Response status:", response.status)

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`)
        }

        const result = await response.json()
        console.log("✅ API result:", result)

        const prediction: PredictionResult = {
          riskScore: result.riskScore,
          confidence: result.confidence,
          features: result.features.map((f: any) => ({
            name: f.name,
            impact: f.impact,
            value: f.value
          })),
          llmExplanation: result.llmExplanation
        }
        setPrediction(prediction)
      } catch (error) {
        console.error("❌ API call failed:", error)
        // Fallback to mock data if API fails
        const mockPrediction: PredictionResult = {
          riskScore: Math.random() * 100,
          confidence: 85 + Math.random() * 15,
          features: [
            { name: "Age", impact: 12.5, value: "68 years" },
            { name: "Lactate Level", impact: 18.3, value: "4.2 mmol/L" },
            { name: "Heart Rate", impact: -5.2, value: "72 bpm" },
          ],
          llmExplanation: `API connection failed. Please ensure the backend is running at http://localhost:8000\n\nError: ${error}`,
        }
        setPrediction(mockPrediction)
      } finally {
        setIsLoading(false)
      }
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0A0E1A] via-[#1a1f35] to-[#0A0E1A]">
      {/* Header */}
      <header className="border-b border-white/10 bg-[#0d1117]/80 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-lg bg-gradient-to-br from-[#4F46E5] to-[#7C3AED]">
                <Activity className="size-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-white">HealthPredict AI</h1>
                <p className="text-sm text-gray-400">Explainable Readmission Risk Assessment</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        {/* Level 1: Data Injection */}
        <Card className="mb-6 border-white/10 bg-[#0d1117]/60 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white">Patient Data Management</CardTitle>
            <CardDescription className="text-gray-400">
              Upload patient data and select a patient for analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" id="file-upload" />
                <label htmlFor="file-upload">
                  <Button
                    variant="outline"
                    className="w-full border-white/20 bg-white/5 text-white hover:bg-white/10"
                    asChild
                  >
                    <span>
                      <Upload className="mr-2 size-4" />
                      Upload Patient CSV
                    </span>
                  </Button>
                </label>
              </div>

              <div className="flex-1">
                <Select value={selectedPatientId} onValueChange={handlePatientSelect} disabled={patients.length === 0}>
                  <SelectTrigger className="border-white/20 bg-white/5 text-white">
                    <SelectValue placeholder="Select patient ID" />
                  </SelectTrigger>
                  <SelectContent className="border-white/20 bg-[#1a1f35]">
                    {patients.map((patient) => (
                      <SelectItem key={patient.id} value={patient.id} className="text-white focus:bg-white/10">
                        {patient.Subject_ID || patient.HADM_ID || patient.id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {selectedPatient && (
              <Collapsible open={isDataPreviewOpen} onOpenChange={setIsDataPreviewOpen}>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" className="w-full text-white hover:bg-white/5">
                    {isDataPreviewOpen ? "Hide" : "Show"} Patient Data (320 features)
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="mt-4">
                  <PatientDataTable data={selectedPatient} />
                </CollapsibleContent>
              </Collapsible>
            )}
          </CardContent>
        </Card>

        {/* Clinical Notes Section */}
        {selectedPatient && selectedPatient.clinical_notes && (
          <Card className="mb-6 border-white/10 bg-[#0d1117]/60 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <svg className="size-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Clinical Notes
                {activeNlpFeatures.length > 0 && (
                  <span className="ml-2 px-2 py-1 bg-yellow-500/20 text-yellow-300 text-xs rounded-full">
                    {activeNlpFeatures.length} NLP features detected
                  </span>
                )}
              </CardTitle>
              <CardDescription className="text-gray-400">
                Nursing and physician notes recorded during ICU stay (before ICU discharge)
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Active NLP Features Pills */}
              {activeNlpFeatures.length > 0 && (
                <div className="mb-4 flex flex-wrap gap-2">
                  {activeNlpFeatures.map((feature) => (
                    <span
                      key={feature}
                      className="px-2 py-1 bg-yellow-500/20 text-yellow-200 text-xs rounded-full border border-yellow-500/30"
                    >
                      {feature.replace('nlp_', '').replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              )}

              {/* Highlighted Notes */}
              <div className="max-h-96 overflow-y-auto rounded-lg bg-white/5 p-4 border border-white/10">
                <div
                  className="whitespace-pre-wrap text-sm text-gray-300 font-mono leading-relaxed"
                  dangerouslySetInnerHTML={{
                    __html: renderHighlightedNotes(selectedPatient.clinical_notes, nlpHighlights)
                  }}
                />
              </div>
              <div className="mt-3 flex items-center gap-2 text-xs text-gray-500">
                <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Highlighted text shows NLP features extracted for risk prediction (hover for feature name)
              </div>
            </CardContent>
          </Card>
        )}

        {/* Level 2: Model Inference & Metrics */}
        {isLoading && (
          <Card className="border-white/10 bg-[#0d1117]/60 backdrop-blur-sm">
            <CardContent className="flex flex-col items-center justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mb-4"></div>
              <p className="text-center text-gray-400">Analyzing patient data with EBM + LLM...</p>
              <p className="text-center text-gray-500 text-sm mt-2">This may take up to 60 seconds for LLM generation</p>
            </CardContent>
          </Card>
        )}

        {prediction && <RiskScoreDisplay riskScore={prediction.riskScore} confidence={prediction.confidence} />}

        {/* Level 3: The Explainable Dashboard */}
        {prediction && (
          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            {/* Left Column: Quantitative Explanation */}
            <QuantitativeExplanation features={prediction.features} />

            {/* Right Column: Qualitative Explanation */}
            <QualitativeExplanation explanation={prediction.llmExplanation} />
          </div>
        )}

        {!prediction && !isLoading && patients.length > 0 && (
          <Card className="border-white/10 bg-[#0d1117]/60 backdrop-blur-sm">
            <CardContent className="flex flex-col items-center justify-center py-12">
              <AlertCircle className="mb-4 size-12 text-gray-500" />
              <p className="text-center text-gray-400">Select a patient to view risk assessment and explanations</p>
            </CardContent>
          </Card>
        )}

        {patients.length === 0 && (
          <Card className="border-white/10 bg-[#0d1117]/60 backdrop-blur-sm">
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Upload className="mb-4 size-12 text-gray-500" />
              <p className="text-center text-gray-400">Upload a CSV file containing patient data to begin</p>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  )
}
