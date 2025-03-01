import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  const { data, column, newType } = await request.json()

  // Convert column type
  const convertedData = { ...data }
  
  try {
    convertedData[column] = convertedData[column].map((value: any) => {
      switch (newType) {
        case 'number':
          return Number(value)
        case 'string':
          return String(value)
        case 'date':
          return new Date(value)
        case 'boolean':
          return Boolean(value)
        default:
          return value
      }
    })
  } catch (error) {
    console.error('Error converting type:', error)
  }

  return NextResponse.json(convertedData)
} 