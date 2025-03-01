'use client'

import { useState } from 'react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { FileUpload } from '@/components/FileUpload'
import { DataPreview } from '@/components/DataPreview'
import { DataCleaning } from '@/components/DataCleaning'
import { BasicVisualization } from '@/components/BasicVisualization'
import { AdvancedAnalysis } from '@/components/AdvancedAnalysis'
import { InteractivePlots } from '@/components/InteractivePlots'
import { ExportData } from '@/components/ExportData'

export default function Home() {
  const [data, setData] = useState(null)

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12 space-y-4">
          <h1 className="text-4xl font-bold text-indigo-600">
            📊 DataViz Pro
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Transform your data into meaningful insights with our modern visualization and analysis tool.
          </p>
        </div>

        {/* File Upload */}
        <FileUpload onDataLoaded={setData} />

        {data && (
          <Tabs defaultValue="preview" className="mt-8">
            <TabsList className="grid grid-cols-6 gap-4">
              <TabsTrigger value="preview">📊 Data Preview</TabsTrigger>
              <TabsTrigger value="cleaning">🧹 Data Cleaning</TabsTrigger>
              <TabsTrigger value="basic-viz">📈 Basic Visualization</TabsTrigger>
              <TabsTrigger value="advanced">🔍 Advanced Analysis</TabsTrigger>
              <TabsTrigger value="interactive">📉 Interactive Plots</TabsTrigger>
              <TabsTrigger value="export">💾 Export</TabsTrigger>
            </TabsList>

            <TabsContent value="preview">
              <DataPreview data={data} />
            </TabsContent>

            <TabsContent value="cleaning">
              <DataCleaning data={data} onDataCleaned={setData} />
            </TabsContent>

            <TabsContent value="basic-viz">
              <BasicVisualization data={data} />
            </TabsContent>

            <TabsContent value="advanced">
              <AdvancedAnalysis data={data} />
            </TabsContent>

            <TabsContent value="interactive">
              <InteractivePlots data={data} />
            </TabsContent>

            <TabsContent value="export">
              <ExportData data={data} />
            </TabsContent>
          </Tabs>
        )}
      </div>
    </main>
  )
} 