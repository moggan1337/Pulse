"""Pulse HTTP Server"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .pipeline import create_pipeline, PulsePipeline
from .collector import Span, Trace, SpanKind
from .context import TraceContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class SpanRequest(BaseModel):
    name: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    service_name: str
    kind: str = "internal"
    duration_ms: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_flag: bool = False
    attributes: Dict[str, Any] = {}
    events: list = []


class TraceRequest(BaseModel):
    trace_id: Optional[str] = None
    spans: list


class AnomalyRequest(BaseModel):
    time_window_minutes: int = 60


class RootCauseRequest(BaseModel):
    time_window_minutes: int = 30


# Global pipeline instance
pipeline: Optional[PulsePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global pipeline
    pipeline = create_pipeline(
        service_name="pulse",
        storage_backend="memory",
    )
    logger.info("Pulse server started")
    yield
    logger.info("Pulse server stopped")


# Create FastAPI app
app = FastAPI(
    title="Pulse",
    description="Distributed Tracing with ML Anomaly Detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "pulse",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# Statistics
@app.get("/stats")
async def stats():
    """Get pipeline statistics"""
    if pipeline:
        return pipeline.get_stats()
    raise HTTPException(status_code=500, detail="Pipeline not initialized")


# Span ingestion
@app.post("/api/v1/spans")
async def ingest_span(span: SpanRequest):
    """Ingest a single span"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    # Convert to internal span
    kind_map = {
        "internal": SpanKind.INTERNAL,
        "server": SpanKind.SERVER,
        "client": SpanKind.CLIENT,
        "producer": SpanKind.PRODUCER,
        "consumer": SpanKind.CONSUMER,
    }
    
    span_obj = Span(
        name=span.name,
        trace_id=span.trace_id or Span.generate_trace_id(),
        span_id=span.span_id or Span.generate_span_id(),
        parent_span_id=span.parent_span_id,
        service_name=span.service_name,
        kind=kind_map.get(span.kind, SpanKind.INTERNAL),
        duration_ms=span.duration_ms,
        error_flag=span.error_flag,
        attributes=span.attributes,
    )
    
    # Set times if provided
    if span.start_time:
        try:
            span_obj.start_time = datetime.fromisoformat(span.start_time)
        except ValueError:
            pass
    
    if span.end_time:
        try:
            span_obj.end_time = datetime.fromisoformat(span.end_time)
        except ValueError:
            pass
    
    pipeline.receive_span(span_obj)
    
    return {"status": "ok", "span_id": span_obj.span_id}


@app.post("/api/v1/traces")
async def ingest_trace(trace: TraceRequest):
    """Ingest a complete trace"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    # Convert spans
    spans = []
    for span_req in trace.spans:
        kind_map = {
            "internal": SpanKind.INTERNAL,
            "server": SpanKind.SERVER,
            "client": SpanKind.CLIENT,
            "producer": SpanKind.PRODUCER,
            "consumer": SpanKind.CONSUMER,
        }
        
        span = Span(
            name=span_req.get("name", "unknown"),
            trace_id=trace.trace_id or span_req.get("trace_id", Span.generate_trace_id()),
            span_id=span_req.get("span_id", Span.generate_span_id()),
            parent_span_id=span_req.get("parent_span_id"),
            service_name=span_req.get("service_name", "unknown"),
            kind=kind_map.get(span_req.get("kind", "internal"), SpanKind.INTERNAL),
            duration_ms=span_req.get("duration_ms", 0),
            error_flag=span_req.get("error_flag", False),
            attributes=span_req.get("attributes", {}),
        )
        spans.append(span)
    
    # Create trace
    trace_obj = Trace(trace_id=trace.trace_id or spans[0].trace_id if spans else Span.generate_trace_id())
    for span in spans:
        trace_obj.add_span(span)
    
    pipeline.receive_trace(trace_obj)
    
    return {"status": "ok", "trace_id": trace_obj.trace_id, "spans": len(spans)}


# Query endpoints
@app.get("/api/v1/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Get a trace by ID"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    trace = pipeline.collector.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return trace.to_dict()


@app.get("/api/v1/traces")
async def list_traces(
    service: Optional[str] = None,
    limit: int = 100,
):
    """List traces with optional filters"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    traces = pipeline.collector.get_traces(service_name=service, limit=limit)
    
    return {
        "traces": [t.get_summary() for t in traces],
        "total": len(traces),
    }


# ML Anomaly Detection
@app.post("/api/v1/analyze/anomalies")
async def analyze_anomalies(request: AnomalyRequest):
    """Analyze traces for anomalies"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not pipeline.anomaly_detector:
        raise HTTPException(status_code=501, detail="ML not enabled")
    
    anomalies = pipeline.analyze_anomalies(request.time_window_minutes)
    
    return {
        "anomalies": anomalies,
        "total": len(anomalies),
        "time_window_minutes": request.time_window_minutes,
    }


# Root Cause Analysis
@app.post("/api/v1/analyze/root-cause")
async def analyze_root_cause(request: RootCauseRequest):
    """Perform root cause analysis"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    result = pipeline.analyze_root_cause(request.time_window_minutes)
    
    if not result:
        return {"message": "No traces found for analysis"}
    
    return result


# Dependency Graph
@app.get("/api/v1/graphs/dependencies")
async def get_dependency_graph(time_window_minutes: int = 60):
    """Get service dependency graph"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    graph_data = pipeline.get_dependency_graph(time_window_minutes)
    
    return graph_data


@app.get("/api/v1/graphs/bottlenecks")
async def get_bottlenecks(time_window_minutes: int = 60):
    """Get identified bottlenecks"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    graph_data = pipeline.get_dependency_graph(time_window_minutes)
    
    return {
        "bottlenecks": graph_data.get("bottlenecks", []),
    }


# SLO Status
@app.get("/api/v1/slo/status")
async def get_slo_status():
    """Get SLO compliance status"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    return pipeline.get_slo_status()


# Alerts
@app.get("/api/v1/alerts")
async def get_alerts(
    severity: Optional[str] = None,
    service: Optional[str] = None,
):
    """Get active alerts"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    from .alerting import AlertSeverity
    
    severity_enum = None
    if severity:
        try:
            severity_enum = AlertSeverity(severity)
        except ValueError:
            pass
    
    alerts = pipeline.get_active_alerts()
    
    # Filter
    if severity_enum:
        alerts = [a for a in alerts if a.severity == severity_enum]
    if service:
        alerts = [a for a in alerts if a.service_name == service]
    
    return {
        "alerts": [
            {
                "alert_id": a.alert_id,
                "title": a.title,
                "severity": a.severity.value,
                "status": a.status.value,
                "service": a.service_name,
                "fired_at": a.fired_at.isoformat(),
                "description": a.description,
            }
            for a in alerts
        ],
        "total": len(alerts),
    }


# Model Training
@app.post("/api/v1/ml/train")
async def train_models():
    """Train ML models on historical data"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not pipeline.anomaly_detector:
        raise HTTPException(status_code=501, detail="ML not enabled")
    
    # Get historical traces
    filter = pipeline.storage.query_traces.__self__._backend.query_traces(
        type('QueryFilter', (), {'limit': 1000, 'offset': 0, 'service_name': None,
                                  'error_only': False, 'trace_id': None,
                                  'start_time': None, 'end_time': None,
                                  'min_duration_ms': None, 'max_duration_ms': None})()
    )
    traces = pipeline.storage.query_traces()
    
    if len(traces) < 10:
        raise HTTPException(status_code=400, detail="Not enough data for training")
    
    result = pipeline.train_models(traces)
    
    return result


# Context Propagation
@app.post("/api/v1/context/extract")
async def extract_context(request: Request):
    """Extract trace context from headers"""
    headers = dict(request.headers)
    
    context = TraceContext.from_w3c_headers(headers)
    
    if not context:
        raise HTTPException(status_code=400, detail="No valid trace context found")
    
    return context.to_dict()


@app.post("/api/v1/context/inject")
async def inject_context(trace_id: str, span_id: str):
    """Generate injected headers for a new context"""
    context = TraceContext(
        trace_id=trace_id,
        span_id=span_id,
    )
    
    headers = context.to_w3c_headers()
    
    return {
        "headers": headers,
        "traceparent": headers.get("traceparent"),
        "tracestate": headers.get("tracestate"),
    }


# Storage stats
@app.get("/api/v1/storage/stats")
async def storage_stats():
    """Get storage statistics"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    return pipeline.storage.get_stats()


# Run server
def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the Pulse server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
