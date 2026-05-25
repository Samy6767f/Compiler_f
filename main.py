import asyncio
import json
import time
import logging
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, AsyncGenerator

from pipeline.orchestrator import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-compiler")

app = FastAPI(title="AI Compiler API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = Pipeline(use_llm=False)
pipeline_llm = Pipeline(use_llm=True)

class CompileRequest(BaseModel):
    prompt: str
    use_llm: bool = False

STAGES = [
    {"name": "intent_extraction", "display": "Intent Extraction"},
    {"name": "system_design", "display": "System Design"},
    {"name": "schema_generation", "display": "Schema Generation"},
    {"name": "validation_repair", "display": "Validation"},
    {"name": "simulation", "display": "Simulation"},
]

tracker_store: Dict[str, 'ProgressTracker'] = {}

class ProgressTracker:
    def __init__(self, request_id: str, stages: list):
        self.request_id = request_id
        self.stages = stages
        self.current_stage = 0
        self.progress = 0
        self.start_time = time.time()
        self.is_complete = False
        self._queue = asyncio.Queue()
        self._result = None
    
    async def update_stage(self, stage_name: str, status: str, details: str = "", error: str = None):
        stage_names = [s['name'] for s in self.stages]
        if stage_name in stage_names:
            self.current_stage = stage_names.index(stage_name) + 1
            self.progress = (self.current_stage / len(self.stages)) * 100
        
        event = {
            "type": "stage_update",
            "stage": stage_name,
            "status": status,
            "progress": self.progress,
            "timestamp": time.time() - self.start_time,
            "details": details,
            "error": error,
        }
        await self._queue.put(event)
        
        if status == "completed" and self.current_stage == len(self.stages):
            self.is_complete = True
            await self._queue.put({"type": "complete", "progress": 100})
    
    async def event_stream(self) -> AsyncGenerator:
        while not self.is_complete or not self._queue.empty():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue
        yield "data: {\"type\": \"close\"}\n\n"

@app.get("/")
async def root():
    return FileResponse("www/index.html")

@app.post("/compile")
async def compile_endpoint(request: CompileRequest, background_tasks: BackgroundTasks):
    request_id = f"req_{int(time.time() * 1000)}"
    
    tracker = ProgressTracker(request_id, STAGES)
    tracker_store[request_id] = tracker
    
    background_tasks.add_task(run_compiler_pipeline, request.prompt, request.use_llm, request_id, tracker)
    
    return JSONResponse({"request_id": request_id})

@app.get("/compile-stream/{request_id}")
async def compile_stream(request_id: str):
    if request_id not in tracker_store:
        return JSONResponse({"error": "Request not found"}, status_code=404)
    
    tracker = tracker_store[request_id]
    return StreamingResponse(
        tracker.event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/compile-result/{request_id}")
async def get_result(request_id: str):
    if request_id not in tracker_store:
        return JSONResponse({"error": "Request not found"}, status_code=404)
    tracker = tracker_store[request_id]
    if not tracker.is_complete:
        return JSONResponse({"error": "Still processing"}, status_code=202)
    return JSONResponse(tracker._result or {"error": "No result"})

async def run_compiler_pipeline(prompt: str, use_llm: bool, request_id: str, tracker: ProgressTracker):
    try:
        await tracker.update_stage("intent_extraction", "started", "Analyzing request...")
        intent = await asyncio.get_event_loop().run_in_executor(None, lambda: pipeline_llm.intent_extractor.extract(prompt) if use_llm else pipeline.intent_extractor.extract(prompt))
        await tracker.update_stage("intent_extraction", "completed", f"Found {len(intent.get('entities', []))} entities")
        
        await tracker.update_stage("system_design", "started", "Designing architecture...")
        design = await asyncio.get_event_loop().run_in_executor(None, lambda: pipeline_llm.system_designer.design(intent) if use_llm else pipeline.system_designer.design(intent))
        await tracker.update_stage("system_design", "completed", f"Designed {len(design.get('pages', []))} pages")
        
        await tracker.update_stage("schema_generation", "started", "Generating schemas...")
        schemas = await asyncio.get_event_loop().run_in_executor(None, lambda: pipeline_llm.schema_generator.generate(design) if use_llm else pipeline.schema_generator.generate(design))
        await tracker.update_stage("schema_generation", "completed", "Schemas generated")
        
        await tracker.update_stage("validation_repair", "started", "Validating...")
        validation_errors = pipeline.validator.validate_cross_layer(design, schemas)
        await tracker.update_stage("validation_repair", "completed", "Validation passed" if not validation_errors else f"{len(validation_errors)} warnings")
        
        await tracker.update_stage("simulation", "started", "Running simulation...")
        simulation = pipeline.simulator.simulate_execution(schemas)
        await tracker.update_stage("simulation", "completed", "Simulation complete")
        
        result = {
            "success": simulation['can_execute'] and len(validation_errors) == 0,
            "request_id": request_id,
            "intent": intent,
            "design": design,
            "schemas": schemas,
            "validation": {"valid": len(validation_errors) == 0, "errors": validation_errors},
            "simulation_result": simulation,
            "metrics": {"stages": {}},
            "latency_ms": 0,
            "stage_errors": []
        }
        
        tracker._result = result
        
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        await tracker.update_stage("intent_extraction", "error", str(e))
        tracker._result = {"error": str(e)}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)