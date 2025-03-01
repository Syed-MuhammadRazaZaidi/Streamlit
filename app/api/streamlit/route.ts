import { NextResponse } from 'next/server'
import { spawn } from 'child_process'

export async function GET() {
  // Start Streamlit server
  const streamlit = spawn('streamlit', ['run', 'streamlit_app.py'])
  
  return NextResponse.json({ status: 'running' })
} 