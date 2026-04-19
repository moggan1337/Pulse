"""OpenTelemetry Trace Collector Module"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import uuid
import hashlib

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

logger = logging.getLogger(__name__)


class TraceState(Enum):
    """Possible states of a trace"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


class SpanKind(Enum):
    """OpenTelemetry Span Kind types"""
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"


@dataclass
class SpanEvent:
    """Represents a span event within a trace"""
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)
    dropped_attributes_count: int = 0


@dataclass
class SpanLink:
    """Links to other traces for context propagation"""
    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """
    Represents a distributed trace span with full OpenTelemetry compatibility.
    
    A span is a named operation representing a single unit of work in a distributed system.
    Spans form a tree structure reflecting the causal relationships between operations.
    """
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    service_name: str = "unknown"
    kind: SpanKind = SpanKind.INTERNAL
    
    # Timing information
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Status and errors
    status_code: str = "OK"
    status_message: str = ""
    error_flag: bool = False
    
    # Attributes and events
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    
    # Resource information
    resource_attributes: Dict[str, str] = field(default_factory=dict)
    instrumentation_library: str = "pulse-collector"
    
    # Sampling information
    sampling_priority: int = 1
    dropped_attributes_count: int = 0
    dropped_events_count: int = 0
    
    def __post_init__(self):
        if self.trace_id is None:
            self.trace_id = self.generate_trace_id()
        if self.span_id is None:
            self.span_id = self.generate_span_id()
    
    @staticmethod
    def generate_trace_id() -> str:
        """Generate a unique 32-character trace ID"""
        return uuid.uuid4().hex
    
    @staticmethod
    def generate_span_id() -> str:
        """Generate a unique 16-character span ID"""
        return uuid.uuid4().hex[:16]
    
    def add_attribute(self, key: str, value: Any) -> None:
        """Add or update an attribute on this span"""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to this span"""
        event = SpanEvent(
            name=name,
            timestamp=datetime.utcnow(),
            attributes=attributes or {}
        )
        self.events.append(event)
    
    def set_status(self, code: str, message: str = "") -> None:
        """Set the span status"""
        self.status_code = code
        self.status_message = message
        if code in ("ERROR", "FAIL"):
            self.error_flag = True
    
    def finish(self, end_time: Optional[datetime] = None) -> None:
        """Mark the span as finished and calculate duration"""
        self.end_time = end_time or datetime.utcnow()
        if self.start_time:
            delta = self.end_time - self.start_time
            self.duration_ms = delta.total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization"""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "service_name": self.service_name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status_code": self.status_code,
            "status_message": self.status_message,
            "error_flag": self.error_flag,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes
                }
                for e in self.events
            ],
            "links": [
                {
                    "trace_id": l.trace_id,
                    "span_id": l.span_id,
                    "attributes": l.attributes
                }
                for l in self.links
            ],
            "resource_attributes": self.resource_attributes,
            "instrumentation_library": self.instrumentation_library,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Span":
        """Create a Span from a dictionary"""
        kind = SpanKind(data.get("kind", "internal"))
        start_time = datetime.fromisoformat(data["start_time"]) if data.get("start_time") else datetime.utcnow()
        end_time = datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
        
        span = cls(
            name=data["name"],
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            service_name=data.get("service_name", "unknown"),
            kind=kind,
            start_time=start_time,
            end_time=end_time,
            status_code=data.get("status_code", "OK"),
            status_message=data.get("status_message", ""),
            attributes=data.get("attributes", {}),
        )
        span.duration_ms = data.get("duration_ms", 0.0)
        span.error_flag = data.get("error_flag", False)
        
        for event_data in data.get("events", []):
            span.events.append(SpanEvent(
                name=event_data["name"],
                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                attributes=event_data.get("attributes", {})
            ))
        
        for link_data in data.get("links", []):
            span.links.append(SpanLink(
                trace_id=link_data["trace_id"],
                span_id=link_data["span_id"],
                attributes=link_data.get("attributes", {})
            ))
        
        return span


@dataclass
class Trace:
    """
    Represents a complete distributed trace containing multiple related spans.
    
    A trace represents the entire journey of a request or operation as it propagates
    through a distributed system. It contains the root span and all child spans.
    """
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
    state: TraceState = TraceState.ACTIVE
    service_graph: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.trace_id is None:
            self.trace_id = Span.generate_trace_id()
    
    def add_span(self, span: Span) -> None:
        """Add a span to this trace"""
        self.spans.append(span)
        self._update_timing()
        self._update_service_graph(span)
    
    def _update_timing(self) -> None:
        """Update trace timing based on spans"""
        if not self.spans:
            return
        
        start_times = [s.start_time for s in self.spans if s.start_time]
        end_times = [s.end_time for s in self.spans if s.end_time]
        
        if start_times:
            self.start_time = min(start_times)
        if end_times:
            self.end_time = max(end_times)
        
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            self.total_duration_ms = delta.total_seconds() * 1000
    
    def _update_service_graph(self, span: Span) -> None:
        """Build service dependency graph from spans"""
        service = span.service_name
        if service not in self.service_graph:
            self.service_graph[service] = []
        
        if span.parent_span_id:
            parent_span = self.get_span_by_id(span.parent_span_id)
            if parent_span:
                parent_service = parent_span.service_name
                if parent_service not in self.service_graph[service]:
                    self.service_graph[service].append(parent_service)
    
    def get_span_by_id(self, span_id: str) -> Optional[Span]:
        """Find a span by its ID"""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None
    
    def get_root_span(self) -> Optional[Span]:
        """Get the root span (no parent)"""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return None
    
    def get_child_spans(self, parent_span_id: str) -> List[Span]:
        """Get all direct child spans of a parent"""
        return [s for s in self.spans if s.parent_span_id == parent_span_id]
    
    def get_spans_by_service(self, service_name: str) -> List[Span]:
        """Get all spans for a specific service"""
        return [s for s in self.spans if s.service_name == service_name]
    
    def calculate_error_rate(self) -> float:
        """Calculate the error rate across all spans"""
        if not self.spans:
            return 0.0
        error_count = sum(1 for s in self.spans if s.error_flag)
        return error_count / len(self.spans)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this trace"""
        services = set(s.service_name for s in self.spans)
        return {
            "trace_id": self.trace_id,
            "total_spans": len(self.spans),
            "services": list(services),
            "total_duration_ms": self.total_duration_ms,
            "error_rate": self.calculate_error_rate(),
            "state": self.state.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization"""
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "state": self.state.value,
            "service_graph": self.service_graph,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trace":
        """Create a Trace from a dictionary"""
        trace = cls(
            trace_id=data["trace_id"],
            state=TraceState(data.get("state", "active")),
            metadata=data.get("metadata", {}),
        )
        
        for span_data in data.get("spans", []):
            trace.spans.append(Span.from_dict(span_data))
        
        if data.get("start_time"):
            trace.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            trace.end_time = datetime.fromisoformat(data["end_time"])
        
        trace.total_duration_ms = data.get("total_duration_ms", 0.0)
        trace.service_graph = data.get("service_graph", {})
        
        return trace


class SpanProcessor:
    """
    Processes and transforms spans before storage.
    
    This class provides hooks for span processing, including attribute enrichment,
    sampling decisions, and filtering.
    """
    
    def __init__(self):
        self.pre_processors: List[Callable[[Span], Span]] = []
        self.post_processors: List[Callable[[Span], Span]] = []
        self.sampling_strategies: Dict[str, float] = {}
        self.attribute_enrichers: List[Callable[[Span], Dict[str, Any]]] = []
    
    def add_pre_processor(self, processor: Callable[[Span], Span]) -> None:
        """Add a pre-processing function"""
        self.pre_processors.append(processor)
    
    def add_post_processor(self, processor: Callable[[Span], Span]) -> None:
        """Add a post-processing function"""
        self.post_processors.append(processor)
    
    def add_attribute_enricher(self, enricher: Callable[[Span], Dict[str, Any]]) -> None:
        """Add an attribute enrichment function"""
        self.attribute_enrichers.append(enricher)
    
    def set_sampling_rate(self, service_name: str, rate: float) -> None:
        """Set sampling rate for a service (0.0 to 1.0)"""
        self.sampling_strategies[service_name] = max(0.0, min(1.0, rate))
    
    def should_sample(self, span: Span) -> bool:
        """Determine if a span should be sampled based on strategy"""
        rate = self.sampling_strategies.get(span.service_name, 1.0)
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        
        # Deterministic sampling based on trace_id hash
        hash_value = int(hashlib.md5(span.trace_id.encode()).hexdigest(), 16)
        return (hash_value % 10000) / 10000.0 < rate
    
    def process(self, span: Span) -> Optional[Span]:
        """Process a span through all pre and post processors"""
        # Check sampling
        if not self.should_sample(span):
            return None
        
        # Run pre-processors
        for processor in self.pre_processors:
            span = processor(span)
            if span is None:
                return None
        
        # Enrich attributes
        for enricher in self.attribute_enrichers:
            extra_attrs = enricher(span)
            span.attributes.update(extra_attrs)
        
        # Run post-processors
        for processor in self.post_processors:
            span = processor(span)
            if span is None:
                return None
        
        return span


class TraceCollector:
    """
    Main collector class for receiving and managing distributed traces.
    
    This collector integrates with OpenTelemetry and provides additional
    functionality for trace aggregation, processing, and analysis.
    
    Features:
    - OpenTelemetry SDK integration
    - Custom span processing pipeline
    - Trace aggregation and grouping
    - Real-time trace streaming
    - Context propagation
    """
    
    def __init__(
        self,
        service_name: str = "pulse-collector",
        enable_otel: bool = True,
        buffer_size: int = 10000,
        flush_interval: float = 5.0,
    ):
        self.service_name = service_name
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Storage
        self.traces: Dict[str, Trace] = {}
        self.pending_spans: Dict[str, List[Span]] = defaultdict(list)
        
        # Processors
        self.span_processor = SpanProcessor()
        self._setup_default_processors()
        
        # OpenTelemetry
        self.otel_provider: Optional[TracerProvider] = None
        if enable_otel and OTEL_AVAILABLE:
            self._setup_opentelemetry()
        
        # Streaming
        self.trace_listeners: List[Callable[[Trace], None]] = []
        self.span_listeners: List[Callable[[Span], None]] = []
        
        # Statistics
        self.stats = {
            "spans_received": 0,
            "spans_processed": 0,
            "spans_dropped": 0,
            "traces_started": 0,
            "traces_completed": 0,
            "traces_errored": 0,
        }
        
        self._running = False
        logger.info(f"TraceCollector initialized for service: {service_name}")
    
    def _setup_default_processors(self) -> None:
        """Setup default span processing rules"""
        # Add timestamp enricher
        self.span_processor.add_attribute_enricher(
            lambda span: {
                "collector.timestamp": datetime.utcnow().isoformat(),
                "collector.version": "1.0.0",
            }
        )
        
        # Add error flag normalizer
        self.span_processor.add_post_processor(self._normalize_error_status)
    
    def _normalize_error_status(self, span: Span) -> Span:
        """Normalize error status across different conventions"""
        if span.status_code in ("ERROR", "FAIL", "UNSET", 2):
            span.error_flag = True
            span.status_code = "ERROR"
        return span
    
    def _setup_opentelemetry(self) -> None:
        """Initialize OpenTelemetry SDK"""
        try:
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.service_name,
                ResourceAttributes.SERVICE_VERSION: "1.0.0",
            })
            
            self.otel_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(self.otel_provider)
            
            logger.info("OpenTelemetry SDK initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry: {e}")
            self.otel_provider = None
    
    def get_tracer(self, name: str = "pulse"):
        """Get an OpenTelemetry tracer instance"""
        if OTEL_AVAILABLE:
            return trace.get_tracer(name)
        return None
    
    def receive_span(self, span: Span) -> None:
        """
        Receive a span from an instrumented service.
        
        This is the main entry point for receiving spans from the OpenTelemetry
        SDK or other instrumentation libraries.
        """
        self.stats["spans_received"] += 1
        
        # Process the span
        processed_span = self.span_processor.process(span)
        if processed_span is None:
            self.stats["spans_dropped"] += 1
            return
        
        self.stats["spans_processed"] += 1
        
        # Add to pending spans
        self.pending_spans[span.trace_id].append(processed_span)
        
        # Notify span listeners
        for listener in self.span_listeners:
            try:
                listener(processed_span)
            except Exception as e:
                logger.error(f"Span listener error: {e}")
        
        # Check if trace is complete
        self._check_trace_completion(span.trace_id)
    
    def receive_trace(self, trace: Trace) -> None:
        """Receive a complete trace"""
        for span in trace.spans:
            self.receive_span(span)
    
    def receive_otel_span(self, span_data: Dict[str, Any]) -> None:
        """Convert and receive an OpenTelemetry span"""
        span = self._convert_otel_span(span_data)
        if span:
            self.receive_span(span)
    
    def _convert_otel_span(self, span_data: Dict[str, Any]) -> Optional[Span]:
        """Convert OpenTelemetry span format to Pulse span"""
        try:
            kind_map = {
                "SERVER": SpanKind.SERVER,
                "CLIENT": SpanKind.CLIENT,
                "PRODUCER": SpanKind.PRODUCER,
                "CONSUMER": SpanKind.CONSUMER,
            }
            
            kind = kind_map.get(span_data.get("kind", ""), SpanKind.INTERNAL)
            
            span = Span(
                name=span_data.get("name", "unknown"),
                trace_id=span_data.get("trace_id", ""),
                span_id=span_data.get("span_id", ""),
                parent_span_id=span_data.get("parent_span_id"),
                service_name=span_data.get("resource", {}).get("service.name", "unknown"),
                kind=kind,
            )
            
            # Parse timestamps
            if span_data.get("start_time"):
                span.start_time = datetime.fromisoformat(
                    span_data["start_time"].replace("Z", "+00:00")
                )
            if span_data.get("end_time"):
                span.end_time = datetime.fromisoformat(
                    span_data["end_time"].replace("Z", "+00:00")
                )
            
            span.attributes = span_data.get("attributes", {})
            span.status_code = span_data.get("status", {}).get("code", "OK")
            
            return span
        except Exception as e:
            logger.error(f"Failed to convert OTel span: {e}")
            return None
    
    def _check_trace_completion(self, trace_id: str) -> None:
        """Check if a trace is complete and finalize it"""
        spans = self.pending_spans.get(trace_id, [])
        if not spans:
            return
        
        # Check for root span completion
        root_span = next((s for s in spans if s.parent_span_id is None), None)
        if root_span and root_span.end_time:
            # All children should be complete
            self._finalize_trace(trace_id)
    
    def _finalize_trace(self, trace_id: str) -> None:
        """Finalize a completed trace"""
        spans = self.pending_spans.pop(trace_id, [])
        if not spans:
            return
        
        trace = Trace(trace_id=trace_id)
        for span in spans:
            trace.add_span(span)
        
        # Determine trace state
        if trace.calculate_error_rate() > 0:
            trace.state = TraceState.ERROR
            self.stats["traces_errored"] += 1
        else:
            trace.state = TraceState.COMPLETED
            self.stats["traces_completed"] += 1
        
        self.traces[trace_id] = trace
        
        # Notify trace listeners
        for listener in self.trace_listeners:
            try:
                listener(trace)
            except Exception as e:
                logger.error(f"Trace listener error: {e}")
        
        logger.debug(f"Trace {trace_id} finalized: {trace.state.value}")
    
    def add_trace_listener(self, listener: Callable[[Trace], None]) -> None:
        """Add a listener for completed traces"""
        self.trace_listeners.append(listener)
    
    def add_span_listener(self, listener: Callable[[Span], None]) -> None:
        """Add a listener for processed spans"""
        self.span_listeners.append(listener)
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID"""
        return self.traces.get(trace_id)
    
    def get_traces(
        self,
        service_name: Optional[str] = None,
        state: Optional[TraceState] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trace]:
        """Query traces with filters"""
        results = list(self.traces.values())
        
        if service_name:
            results = [
                t for t in results
                if any(s.service_name == service_name for s in t.spans)
            ]
        
        if state:
            results = [t for t in results if t.state == state]
        
        if start_time:
            results = [t for t in results if t.start_time and t.start_time >= start_time]
        
        if end_time:
            results = [t for t in results if t.end_time and t.end_time <= end_time]
        
        # Sort by start time descending
        results.sort(key=lambda t: t.start_time or datetime.min, reverse=True)
        
        return results[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics"""
        return {
            **self.stats,
            "pending_traces": len(self.pending_spans),
            "stored_traces": len(self.traces),
            "buffer_size": self.buffer_size,
        }
    
    def flush(self) -> None:
        """Force flush of all pending data"""
        for trace_id in list(self.pending_spans.keys()):
            self._finalize_trace(trace_id)
        logger.info("Collector flushed")
    
    def clear(self) -> None:
        """Clear all stored traces"""
        self.traces.clear()
        self.pending_spans.clear()
        logger.info("Collector cleared")


class CollectorServer:
    """
    HTTP server for receiving traces from remote services.
    
    Provides endpoints for:
    - POST /spans - Receive single span
    - POST /traces - Receive complete trace
    - GET /traces/:id - Get trace by ID
    - GET /stats - Get collector statistics
    """
    
    def __init__(self, collector: TraceCollector, host: str = "0.0.0.0", port: int = 4318):
        self.collector = collector
        self.host = host
        self.port = port
        self.app = None  # Will be set by aiohttp
    
    async def handle_spans(self, request) -> Dict[str, Any]:
        """Handle incoming spans"""
        data = await request.json()
        
        if isinstance(data, list):
            for span_data in data:
                self.collector.receive_otel_span(span_data)
        else:
            self.collector.receive_otel_span(data)
        
        return {"status": "ok", "spans_received": 1 if isinstance(data, dict) else len(data)}
    
    async def handle_traces(self, request) -> Dict[str, Any]:
        """Handle incoming traces"""
        data = await request.json()
        trace = Trace.from_dict(data)
        self.collector.receive_trace(trace)
        
        return {"status": "ok", "trace_id": trace.trace_id}
    
    async def handle_get_trace(self, request, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific trace"""
        trace = self.collector.get_trace(trace_id)
        if trace:
            return trace.to_dict()
        return None
    
    async def handle_stats(self, request) -> Dict[str, Any]:
        """Get collector statistics"""
        return self.collector.get_stats()
    
    async def handle_health(self, request) -> Dict[str, str]:
        """Health check endpoint"""
        return {"status": "healthy"}
    
    def setup_routes(self, app) -> None:
        """Setup HTTP routes"""
        app.router.add_post("/v1/spans", self.handle_spans)
        app.router.add_post("/v1/traces", self.handle_traces)
        app.router.add_get("/v1/traces/{trace_id}", self.handle_get_trace)
        app.router.add_get("/v1/stats", self.handle_stats)
        app.router.add_get("/health", self.handle_health)


# Convenience function for quick setup
def create_collector(
    service_name: str = "pulse-collector",
    enable_otel: bool = True,
) -> TraceCollector:
    """Create a configured trace collector"""
    return TraceCollector(
        service_name=service_name,
        enable_otel=enable_otel,
    )
