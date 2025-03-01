'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Progress } from './ui/progress'
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs'
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from './ui/table'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select'
import { Badge } from './ui/badge'
import { Alert, AlertDescription } from './ui/alert'
import { Loader2 } from 'lucide-react'

interface DataCleaningProps {
  data: any
  onDataCleaned: (cleanedData: any) => void
}

export function DataCleaning({ data, onDataCleaned }: DataCleaningProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [selectedColumns, setSelectedColumns] = useState<string[]>([])
  const [cleaningOptions, setCleaningOptions] = useState({
    removeDuplicates: false,
    removeNulls: false,
    stripWhitespace: false,
    convertTypes: false
  })

  // Get column statistics
  const columnStats = Object.entries(data).map(([column, values]) => ({
    name: column,
    type: typeof values[0],
    nullCount: values.filter((v: any) => v === null || v === '').length,
    uniqueCount: new Set(values).size,
    total: values.length
  }))

  const handleTypeConversion = async (column: string, newType: string) => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/convert-type', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data,
          column,
          newType
        })
      })
      const convertedData = await response.json()
      onDataCleaned(convertedData)
    } catch (error) {
      console.error('Error converting type:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleCleanData = async () => {
    setIsLoading(true)
    setProgress(0)
    
    try {
      // Simulate progress
      const interval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90))
      }, 200)

      const response = await fetch('/api/clean-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data,
          options: cleaningOptions,
          columns: selectedColumns
        })
      })

      clearInterval(interval)
      setProgress(100)

      const cleanedData = await response.json()
      onDataCleaned(cleanedData)
    } catch (error) {
      console.error('Error cleaning data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="quick-clean">
        <TabsList>
          <TabsTrigger value="quick-clean">Quick Clean</TabsTrigger>
          <TabsTrigger value="column-types">Column Types</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>

        <TabsContent value="quick-clean">
          <Card>
            <CardHeader>
              <CardTitle>Quick Clean Options</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="removeDuplicates"
                    checked={cleaningOptions.removeDuplicates}
                    onChange={(e) => setCleaningOptions(prev => ({
                      ...prev,
                      removeDuplicates: e.target.checked
                    }))}
                    className="rounded border-gray-300"
                  />
                  <label htmlFor="removeDuplicates" className="text-sm">
                    Remove duplicate rows
                  </label>
                </div>

                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="removeNulls"
                    checked={cleaningOptions.removeNulls}
                    onChange={(e) => setCleaningOptions(prev => ({
                      ...prev,
                      removeNulls: e.target.checked
                    }))}
                    className="rounded border-gray-300"
                  />
                  <label htmlFor="removeNulls" className="text-sm">
                    Remove rows with null values
                  </label>
                </div>

                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="stripWhitespace"
                    checked={cleaningOptions.stripWhitespace}
                    onChange={(e) => setCleaningOptions(prev => ({
                      ...prev,
                      stripWhitespace: e.target.checked
                    }))}
                    className="rounded border-gray-300"
                  />
                  <label htmlFor="stripWhitespace" className="text-sm">
                    Strip whitespace from text
                  </label>
                </div>

                <Button 
                  onClick={handleCleanData}
                  disabled={isLoading}
                  className="w-full mt-4"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Cleaning...
                    </>
                  ) : (
                    'Clean Data'
                  )}
                </Button>

                {isLoading && (
                  <Progress value={progress} className="w-full" />
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="column-types">
          <Card>
            <CardHeader>
              <CardTitle>Column Type Conversion</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Column</TableHead>
                    <TableHead>Current Type</TableHead>
                    <TableHead>Convert To</TableHead>
                    <TableHead>Null Count</TableHead>
                    <TableHead>Unique Values</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {columnStats.map((col) => (
                    <TableRow key={col.name}>
                      <TableCell>{col.name}</TableCell>
                      <TableCell>
                        <Badge variant="secondary">
                          {col.type}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Select
                          onValueChange={(value) => handleTypeConversion(col.name, value)}
                        >
                          <SelectTrigger className="w-32">
                            <SelectValue placeholder="Convert to..." />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="string">Text</SelectItem>
                            <SelectItem value="number">Number</SelectItem>
                            <SelectItem value="date">Date</SelectItem>
                            <SelectItem value="boolean">Boolean</SelectItem>
                          </SelectContent>
                        </Select>
                      </TableCell>
                      <TableCell>
                        <Badge variant={col.nullCount > 0 ? "destructive" : "secondary"}>
                          {col.nullCount}
                        </Badge>
                      </TableCell>
                      <TableCell>{col.uniqueCount}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="advanced">
          <Card>
            <CardHeader>
              <CardTitle>Advanced Cleaning Options</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium mb-2">Select Columns to Clean</h3>
                  <div className="flex flex-wrap gap-2">
                    {Object.keys(data).map((column) => (
                      <Badge
                        key={column}
                        variant={selectedColumns.includes(column) ? "default" : "outline"}
                        className="cursor-pointer"
                        onClick={() => {
                          setSelectedColumns(prev => 
                            prev.includes(column) 
                              ? prev.filter(c => c !== column)
                              : [...prev, column]
                          )
                        }}
                      >
                        {column}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Add more advanced cleaning options here */}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Summary Card */}
      <Card>
        <CardHeader>
          <CardTitle>Cleaning Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertDescription>
              Total Rows: {data[Object.keys(data)[0]].length}
              <br />
              Total Columns: {Object.keys(data).length}
              <br />
              Null Values: {columnStats.reduce((acc, col) => acc + col.nullCount, 0)}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    </div>
  )
} 