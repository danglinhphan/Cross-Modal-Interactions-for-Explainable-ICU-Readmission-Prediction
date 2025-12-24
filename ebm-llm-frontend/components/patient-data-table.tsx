import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { PatientData } from "@/types/patient"

interface PatientDataTableProps {
  data: PatientData
}

export function PatientDataTable({ data }: PatientDataTableProps) {
  // Exclude id and clinical_notes (clinical_notes has its own section)
  const entries = Object.entries(data).filter(([key]) =>
    key !== "id" && key !== "clinical_notes" && key !== "CLEAN_TEXT"
  )

  return (
    <ScrollArea className="h-96 rounded-lg border border-white/10 bg-[#0A0E1A]">
      <Table>
        <TableHeader>
          <TableRow className="border-white/10 hover:bg-transparent">
            <TableHead className="text-gray-400">Feature</TableHead>
            <TableHead className="text-gray-400">Value</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {entries.map(([key, value], index) => (
            <TableRow key={index} className="border-white/10 hover:bg-white/5">
              <TableCell className="font-medium text-white">{key}</TableCell>
              <TableCell className="text-gray-300">{String(value)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </ScrollArea>
  )
}
