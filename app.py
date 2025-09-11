from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import asyncio
from datetime import datetime
import uuid
import json
from pathlib import Path

# Import your crew components
from crew import InvestmentCrew, get_chromarag

app = FastAPI(title="AlphaAgent API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store analysis jobs
analysis_jobs: Dict[str, Dict[str, Any]] = {}

class StockAnalysisRequest(BaseModel):
    stock_ticker: str
    include_uploaded_docs: bool = False

class AnalysisStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "AlphaAgent API is running", "version": "1.0.0"}

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a financial document for analysis"""
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Create directory if it doesn't exist
        upload_dir = Path("assets/rag_assets")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file with unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = upload_dir / filename
        
        content = await file.read()
        
        # Handle text files
        if file_extension == '.txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.decode('utf-8'))
        else:
            with open(file_path, 'wb') as f:
                f.write(content)
        
        # Reinitialize ChromaDB with new document (only if it's a PDF)
        if file_extension == '.pdf':
            get_chromarag(force_reinit=True)
        
        return {
            "success": True,
            "filename": filename,
            "message": "Document uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-stock")
async def analyze_stock(request: StockAnalysisRequest, background_tasks: BackgroundTasks):
    """Start stock analysis job"""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        analysis_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Analysis job created",
            "stock_ticker": request.stock_ticker,
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None
        }
        
        # Start analysis in background
        background_tasks.add_task(
            run_analysis,
            job_id,
            request.stock_ticker,
            request.include_uploaded_docs
        )
        
        return {
            "job_id": job_id,
            "message": f"Analysis started for {request.stock_ticker}",
            "status": "pending"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_analysis(job_id: str, stock_ticker: str, include_uploaded_docs: bool):
    """Run the actual stock analysis"""
    try:
        print(f"Starting analysis for {stock_ticker} (Job ID: {job_id})")
        
        # Update status
        analysis_jobs[job_id]["status"] = "processing"
        analysis_jobs[job_id]["progress"] = 10
        analysis_jobs[job_id]["message"] = "Initializing analysis agents..."
        
        # Initialize ChromaDB if needed and documents are uploaded
        if include_uploaded_docs:
            analysis_jobs[job_id]["progress"] = 20
            analysis_jobs[job_id]["message"] = "Loading uploaded documents..."
            chromarag = get_chromarag()
            if chromarag is None:
                analysis_jobs[job_id]["message"] = "No documents found, proceeding without document analysis..."
                print("No documents found for ChromaDB")
        
        # Prepare inputs with the actual stock ticker
        inputs = {
            'topic': f'give me report for {stock_ticker}',
            'stock': stock_ticker  # Pass the stock ticker explicitly
        }
        print(f"Inputs prepared: {inputs}")
        
        # Update progress
        analysis_jobs[job_id]["progress"] = 30
        analysis_jobs[job_id]["message"] = "Starting fundamental analysis..."
        print("Creating crew...")
        
        # Create an instance of InvestmentCrew and set the stock BEFORE creating crew
        investment_crew = InvestmentCrew()
        investment_crew.stock = stock_ticker
        InvestmentCrew.stock = stock_ticker  # Also set the class variable
        print(f"Stock ticker set to: {investment_crew.stock} (class: {InvestmentCrew.stock})")
        
        # Run the crew analysis
        crew = investment_crew.crew()
        print("Crew created, starting kickoff...")
        
        analysis_jobs[job_id]["progress"] = 50
        analysis_jobs[job_id]["message"] = "Performing valuation analysis..."
        
        result = crew.kickoff(inputs=inputs)
        print(f"Analysis complete. Result: {result}")
        
        analysis_jobs[job_id]["progress"] = 80
        analysis_jobs[job_id]["message"] = "Analyzing market sentiment..."
        
        # Parse the result - just return the summary
        analysis_result = {
            "stock_ticker": stock_ticker,
            "timestamp": datetime.now().isoformat(),
            "summary": str(result)  # Just the summary, nothing else
        }
        
        # Update job status
        analysis_jobs[job_id]["status"] = "completed"
        analysis_jobs[job_id]["progress"] = 100
        analysis_jobs[job_id]["message"] = "Analysis completed successfully"
        analysis_jobs[job_id]["result"] = analysis_result
        print(f"Analysis completed successfully for {stock_ticker}")
        
    except Exception as e:
        import traceback
        detailed_error = traceback.format_exc()
        print(f"ERROR in analysis for {stock_ticker}:")
        print(detailed_error)
        
        analysis_jobs[job_id]["status"] = "failed"
        analysis_jobs[job_id]["error"] = str(e)
        analysis_jobs[job_id]["message"] = f"Analysis failed: {str(e)}"

@app.get("/api/analysis-status/{job_id}")
async def get_analysis_status(job_id: str):
    """Get the status of an analysis job"""
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = analysis_jobs[job_id]
    return AnalysisStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        result=job.get("result"),
        error=job.get("error")
    )

@app.get("/api/analysis-stream/{job_id}")
async def stream_analysis_status(job_id: str):
    """Stream analysis status updates using Server-Sent Events"""
    async def generate():
        while True:
            if job_id not in analysis_jobs:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
                
            job = analysis_jobs[job_id]
            data = {
                "status": job["status"],
                "progress": job["progress"],
                "message": job["message"],
                "result": job.get("result"),
                "error": job.get("error")
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
            if job["status"] in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/recent-analyses")
async def get_recent_analyses():
    """Get list of recent analysis jobs"""
    recent = []
    for job_id, job in analysis_jobs.items():
        recent.append({
            "job_id": job_id,
            "stock_ticker": job.get("stock_ticker"),
            "status": job["status"],
            "created_at": job.get("created_at"),
            "message": job["message"]
        })
    
    # Sort by created_at and return last 10
    recent.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return recent[:10]

@app.delete("/api/clear-documents")
async def clear_documents():
    """Clear all uploaded documents"""
    try:
        upload_dir = Path("assets/rag_assets")
        if upload_dir.exists():
            for file in upload_dir.glob("*"):
                if file.is_file():
                    file.unlink()
        
        return {"success": True, "message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use PORT env variable if available
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)