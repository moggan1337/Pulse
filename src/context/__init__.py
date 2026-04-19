"""Distributed Context Propagation Module"""

import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)


# W3C Trace Context constants
TRACE_PARENT_VERSION = "00"
TRACE_STATE_VERSION = "00"


@dataclass
class TraceContext:
    """
    Represents a distributed trace context for context propagation.
    
    Implements W3C Trace Context standard for propagating trace information
    across service boundaries.
    """
    trace_id: str
    span_id: str
    trace_flags: str = "01"  # 01 = sampled, 00 = not sampled
    
    # Trace state for additional context
    trace_state: Dict[str, str] = field(default_factory=dict)
    
    # Additional baggage
    baggage: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    service_name: Optional[str] = None
    operation_name: Optional[str] = None
    
    @property
    def is_sampled(self) -> bool:
        """Check if trace is sampled"""
        return self.trace_flags == "01"
    
    def to_traceparent(self) -> str:
        """
        Convert to W3C Trace Parent header format.
        
        Format: version-trace_id-span_id-trace_flags
        Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
        """
        return f"{TRACE_PARENT_VERSION}-{self.trace_id}-{self.span_id}-{self.trace_flags}"
    
    @classmethod
    def from_traceparent(cls, traceparent: str) -> Optional["TraceContext"]:
        """
        Parse W3C Trace Parent header.
        
        Args:
            traceparent: Trace parent header value
        
        Returns:
            TraceContext or None if parsing fails
        """
        try:
            parts = traceparent.split("-")
            if len(parts) != 4:
                return None
            
            version, trace_id, span_id, trace_flags = parts
            
            # Validate versions
            if version != TRACE_PARENT_VERSION:
                logger.warning(f"Unknown trace parent version: {version}")
            
            # Validate trace_id (32 hex chars)
            if len(trace_id) != 32:
                return None
            
            # Validate span_id (16 hex chars)
            if len(span_id) != 16:
                return None
            
            # Validate trace_flags (2 hex chars)
            if len(trace_flags) != 2:
                return None
            
            return cls(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=trace_flags,
            )
        except Exception as e:
            logger.error(f"Failed to parse traceparent: {e}")
            return None
    
    def to_tracestate(self) -> str:
        """
        Convert to W3C Trace State header format.
        
        Format: key1=value1,key2=value2
        """
        pairs = []
        for key, value in self.trace_state.items():
            # Values with commas or equals need to be quoted
            if "," in value or "=" in value:
                value = f'"{value}"'
            pairs.append(f"{key}={value}")
        return ",".join(pairs)
    
    @classmethod
    def from_tracestate(cls, tracestate: str) -> Dict[str, str]:
        """Parse W3C Trace State header"""
        result = {}
        
        try:
            # Split by comma, but respect quoted values
            parts = _split_tracestate(tracestate)
            
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    # Remove quotes from value
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    result[key.strip()] = value.strip()
        except Exception as e:
            logger.error(f"Failed to parse tracestate: {e}")
        
        return result
    
    def to_w3c_headers(self) -> Dict[str, str]:
        """Convert to standard W3C headers"""
        headers = {
            "traceparent": self.to_traceparent(),
        }
        
        if self.trace_state:
            headers["tracestate"] = self.to_tracestate()
        
        return headers
    
    @classmethod
    def from_w3c_headers(cls, headers: Dict[str, str]) -> Optional["TraceContext"]:
        """Create from W3C headers"""
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None
        
        context = cls.from_traceparent(traceparent)
        
        if not context:
            return None
        
        # Parse tracestate if present
        tracestate = headers.get("tracestate")
        if tracestate:
            context.trace_state = cls.from_tracestate(tracestate)
        
        return context
    
    def add_to_trace_state(self, key: str, value: str) -> None:
        """Add a key-value pair to trace state"""
        self.trace_state[key] = value
    
    def get_from_trace_state(self, key: str) -> Optional[str]:
        """Get a value from trace state"""
        return self.trace_state.get(key)
    
    def set_baggage(self, key: str, value: str) -> None:
        """Set a baggage item"""
        self.baggage[key] = value
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get a baggage item"""
        return self.baggage.get(key)
    
    def get_all_baggage(self) -> Dict[str, str]:
        """Get all baggage items"""
        return self.baggage.copy()
    
    def inject_baggage(self, headers: Dict[str, str]) -> None:
        """Inject baggage into headers"""
        if self.baggage:
            headers[" baggage"] = json.dumps(self.baggage)
    
    @classmethod
    def extract_baggage(cls, headers: Dict[str, str]) -> Dict[str, str]:
        """Extract baggage from headers"""
        baggage_header = headers.get("baggage") or headers.get(" baggage")
        if not baggage_header:
            return {}
        
        try:
            return json.loads(baggage_header)
        except json.JSONDecodeError:
            return {}
    
    def create_child_context(self, span_id: str, operation_name: Optional[str] = None) -> "TraceContext":
        """
        Create a child context for a downstream call.
        
        Args:
            span_id: New span ID for the child
            operation_name: Name of the operation
        
        Returns:
            New TraceContext for the child span
        """
        child = TraceContext(
            trace_id=self.trace_id,
            span_id=span_id,
            trace_flags=self.trace_flags,
            trace_state=self.trace_state.copy(),
            baggage=self.baggage.copy(),
            service_name=self.service_name,
            operation_name=operation_name,
        )
        return child
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_flags": self.trace_flags,
            "trace_state": self.trace_state,
            "baggage": self.baggage,
            "service_name": self.service_name,
            "operation_name": self.operation_name,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceContext":
        """Create from dictionary"""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            trace_flags=data.get("trace_flags", "01"),
            trace_state=data.get("trace_state", {}),
            baggage=data.get("baggage", {}),
            service_name=data.get("service_name"),
            operation_name=data.get("operation_name"),
        )
    
    @classmethod
    def generate_new(
        cls,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        sampled: bool = True,
    ) -> "TraceContext":
        """Generate a new trace context"""
        import uuid
        
        return cls(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            trace_flags="01" if sampled else "00",
            service_name=service_name,
            operation_name=operation_name,
        )


def _split_tracestate(tracestate: str) -> List[str]:
    """Split tracestate header respecting quoted values"""
    parts = []
    current = ""
    in_quotes = False
    
    for char in tracestate:
        if char == '"':
            in_quotes = not in_quotes
            current += char
        elif char == "," and not in_quotes:
            parts.append(current)
            current = ""
        else:
            current += char
    
    if current:
        parts.append(current)
    
    return parts


class ContextPropagator:
    """
    Handles distributed context propagation across service boundaries.
    
    Supports multiple propagation formats:
    - W3C Trace Context (default)
    - B3 Propagation (Zipkin style)
    - Custom formats
    """
    
    def __init__(self):
        self.context_stack: Dict[str, TraceContext] = {}
        self.propagation_format = "w3c"
    
    def set_format(self, format: str) -> None:
        """Set propagation format"""
        self.propagation_format = format
    
    def inject_context(
        self,
        context: TraceContext,
        carrier: Dict[str, str],
        format: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Inject context into a carrier (e.g., HTTP headers, message metadata).
        
        Args:
            context: TraceContext to inject
            carrier: Dictionary to inject into
            format: Propagation format (w3c, b3, etc.)
        
        Returns:
            Updated carrier
        """
        fmt = format or self.propagation_format
        
        if fmt == "w3c":
            carrier.update(context.to_w3c_headers())
            context.inject_baggage(carrier)
        elif fmt == "b3":
            self._inject_b3(context, carrier)
        else:
            logger.warning(f"Unknown propagation format: {fmt}")
        
        return carrier
    
    def extract_context(
        self,
        carrier: Dict[str, str],
        format: Optional[str] = None,
    ) -> Optional[TraceContext]:
        """
        Extract context from a carrier.
        
        Args:
            carrier: Dictionary containing trace context
            format: Propagation format
        
        Returns:
            Extracted TraceContext or None
        """
        fmt = format or self.propagation_format
        
        if fmt == "w3c":
            context = TraceContext.from_w3c_headers(carrier)
            if context:
                context.baggage = TraceContext.extract_baggage(carrier)
            return context
        elif fmt == "b3":
            return self._extract_b3(carrier)
        else:
            logger.warning(f"Unknown propagation format: {fmt}")
            return None
    
    def _inject_b3(self, context: TraceContext, carrier: Dict[str, str]) -> None:
        """Inject B3 format headers"""
        # B3 Single Header: {TraceId}-{SpanId}-{SamplingState}-{ParentSpanId}-{Flags}
        b3_single = f"{context.trace_id}-{context.span_id}-{context.trace_flags}"
        carrier["b3"] = b3_single
        
        # Individual headers
        carrier["X-B3-TraceId"] = context.trace_id
        carrier["X-B3-SpanId"] = context.span_id
        carrier["X-B3-Sampled"] = "1" if context.is_sampled else "0"
    
    def _extract_b3(self, carrier: Dict[str, str]) -> Optional[TraceContext]:
        """Extract B3 format headers"""
        # Try single header first
        b3_single = carrier.get("b3")
        if b3_single:
            parts = b3_single.split("-")
            if len(parts) >= 2:
                trace_id = parts[0]
                span_id = parts[1]
                flags = "01" if len(parts) > 2 and parts[2] == "1" else "00"
                
                return TraceContext(
                    trace_id=trace_id,
                    span_id=span_id,
                    trace_flags=flags,
                )
        
        # Try individual headers
        trace_id = carrier.get("X-B3-TraceId")
        span_id = carrier.get("X-B3-SpanId")
        
        if trace_id and span_id:
            sampled = carrier.get("X-B3-Sampled", "1") == "1"
            return TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags="01" if sampled else "00",
            )
        
        return None
    
    def push_context(self, trace_id: str, context: TraceContext) -> None:
        """Push a context onto the stack"""
        self.context_stack[trace_id] = context
    
    def pop_context(self, trace_id: str) -> Optional[TraceContext]:
        """Pop a context from the stack"""
        return self.context_stack.pop(trace_id, None)
    
    def get_active_context(self, trace_id: str) -> Optional[TraceContext]:
        """Get the active context for a trace"""
        return self.context_stack.get(trace_id)


class MultiPropagator:
    """
    Propagator that supports multiple formats simultaneously.
    
    Useful for transitioning between formats or supporting multiple tracers.
    """
    
    def __init__(self):
        self.propagators: Dict[str, ContextPropagator] = {
            "w3c": ContextPropagator(),
            "b3": ContextPropagator(),
        }
        self.primary_format = "w3c"
    
    def inject(
        self,
        context: TraceContext,
        carrier: Dict[str, str],
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Inject context using multiple formats.
        
        Args:
            context: TraceContext to inject
            carrier: Carrier dictionary
            formats: List of formats to use (defaults to all)
        
        Returns:
            Updated carrier with headers from all formats
        """
        formats_to_use = formats or list(self.propagators.keys())
        
        for fmt in formats_to_use:
            if fmt in self.propagators:
                self.propagators[fmt].inject_context(context, carrier, fmt)
        
        return carrier
    
    def extract(
        self,
        carrier: Dict[str, str],
        formats: Optional[List[str]] = None,
    ) -> Optional[TraceContext]:
        """
        Extract context trying multiple formats.
        
        Tries formats in order until one succeeds.
        """
        formats_to_try = formats or [self.primary_format] + [
            f for f in self.propagators.keys() if f != self.primary_format
        ]
        
        for fmt in formats_to_try:
            if fmt in self.propagators:
                context = self.propagators[fmt].extract_context(carrier, fmt)
                if context:
                    return context
        
        return None


# HTTP-specific context propagation
class HTTPContextPropagator(ContextPropagator):
    """
    HTTP-specific context propagation.
    
    Handles HTTP headers and automatically manages context
    for incoming and outgoing HTTP requests.
    """
    
    def extract_from_request(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Extract context from HTTP request headers"""
        # Normalize header names to lowercase
        normalized = {k.lower(): v for k, v in headers.items()}
        return self.extract_context(normalized)
    
    def inject_into_request(
        self,
        context: TraceContext,
        headers: Dict[str, str],
    ) -> Dict[str, str]:
        """Inject context into HTTP request headers"""
        return self.inject_context(context, headers)
    
    def inject_into_response(
        self,
        context: TraceContext,
        headers: Dict[str, str],
    ) -> Dict[str, str]:
        """Inject context into HTTP response headers"""
        return self.inject_context(context, headers)


# gRPC-specific context propagation
class GRPCContextPropagator(ContextPropagator):
    """
    gRPC-specific context propagation.
    
    Handles gRPC metadata for context propagation.
    """
    
    def extract_from_metadata(self, metadata: Dict[str, Any]) -> Optional[TraceContext]:
        """Extract context from gRPC metadata"""
        # Convert metadata to dict format
        carrier = {}
        for key, value in metadata.items():
            if isinstance(value, (str, bytes)):
                carrier[key] = value
        
        return self.extract_context(carrier)
    
    def inject_into_metadata(
        self,
        context: TraceContext,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inject context into gRPC metadata"""
        carrier = {}
        self.inject_context(context, carrier)
        
        # Convert back to metadata format
        for key, value in carrier.items():
            metadata[key] = value
        
        return metadata


# Message queue context propagation
class MessageContextPropagator(ContextPropagator):
    """
    Message queue-specific context propagation.
    
    Handles message headers for Kafka, RabbitMQ, etc.
    """
    
    def extract_from_message(self, message: Dict[str, Any]) -> Optional[TraceContext]:
        """Extract context from message headers"""
        headers = message.get("headers", {})
        
        # Handle different header formats
        if isinstance(headers, dict):
            # Normalize bytes values to strings
            normalized = {}
            for key, value in headers.items():
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8")
                    except UnicodeDecodeError:
                        value = base64.b64encode(value).decode()
                normalized[key] = value
            return self.extract_context(normalized)
        
        return None
    
    def inject_into_message(
        self,
        context: TraceContext,
        message: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inject context into message headers"""
        if "headers" not in message:
            message["headers"] = {}
        
        self.inject_context(context, message["headers"])
        return message


# Context decorator for automatic propagation
def with_context propagation(func):
    """
    Decorator to automatically propagate context through a function.
    
    Usage:
        @with_context propagation
        def my_function(context, arg1, arg2):
            # context is automatically passed
            pass
    """
    def wrapper(context: TraceContext, *args, **kwargs):
        # Inject context into kwargs for the function
        kwargs["_context"] = context
        return func(context, *args, **kwargs)
    
    return wrapper


# Utility functions
def create_propagator(format: str = "w3c") -> ContextPropagator:
    """Create a configured context propagator"""
    if format == "http":
        return HTTPContextPropagator()
    elif format == "grpc":
        return GRPCContextPropagator()
    elif format == "message":
        return MessageContextPropagator()
    else:
        return ContextPropagator()


def create_multi_propagator() -> MultiPropagator:
    """Create a multi-format propagator"""
    return MultiPropagator()
