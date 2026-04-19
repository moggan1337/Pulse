"""Pulse - Distributed Tracing with ML Anomaly Detection"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .collector import TraceCollector, Trace, Span
from .storage import TraceStorage, StorageConfig, QueryFilter
from .ml import AnomalyDetector, create_detector, generate_synthetic_traces
from .analysis import RootCauseAnalyzer, create_analyzer
from .alerting import PredictiveAlerter, create_alerter, Alert, AlertSeverity
from .graphs import DependencyGraphGenerator, create_graph_generator
from .slo import SLOTracker, SLOTarget, SLOCompliance, create_slo_tracker
from .context import TraceContext, ContextPropagator, create_propagator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PulsePipeline:
    """
    Main pipeline integrating all Pulse components.
    
    Provides a unified interface for:
    - Trace collection
    - Storage and retrieval
    - ML-based anomaly detection
    - Root cause analysis
    - Predictive alerting
    - Dependency graph generation
    - SLO tracking
    - Context propagation
    """
    
    def __init__(
        self,
        service_name: str = "pulse",
        storage_backend: str = "memory",
        enable_ml: bool = True,
        enable_alerting: bool = True,
    ):
        self.service_name = service_name
        self.started_at = datetime.utcnow()
        
        # Initialize components
        logger.info("Initializing Pulse pipeline...")
        
        # Collector
        self.collector = TraceCollector(service_name=service_name)
        
        # Storage
        storage_config = StorageConfig(backend=storage_backend)
        self.storage = TraceStorage(storage_config)
        
        # ML Anomaly Detection
        self.anomaly_detector: Optional[AnomalyDetector] = None
        if enable_ml:
            try:
                self.anomaly_detector = create_detector()
                self.anomaly_detector.initialize()
                logger.info("ML anomaly detection enabled")
            except ImportError as e:
                logger.warning(f"ML libraries not available: {e}")
        
        # Root Cause Analysis
        self.root_cause_analyzer = create_analyzer()
        
        # Alerting
        self.alerter: Optional[PredictiveAlerter] = None
        if enable_alerting:
            self.alerter = create_alerter()
            logger.info("Predictive alerting enabled")
        
        # Dependency Graph
        self.graph_generator = create_graph_generator()
        
        # SLO Tracking
        self.slo_tracker = create_slo_tracker()
        
        # Context Propagation
        self.context_propagator = create_propagator()
        
        # Setup collector listeners
        self._setup_listeners()
        
        logger.info(f"Pulse pipeline initialized for {service_name}")
    
    def _setup_listeners(self) -> None:
        """Setup collectors for automatic processing"""
        
        def on_trace_complete(trace: Trace):
            """Process completed trace"""
            try:
                # Store trace
                self.storage.store_trace_sync(trace.to_dict())
                
                # Run anomaly detection
                if self.anomaly_detector:
                    score = self.anomaly_detector.detect(trace.to_dict())
                    if score.is_anomaly:
                        logger.warning(f"Anomaly detected in trace {trace.trace_id}: {score.normalized:.2f}")
                
                # Update SLO tracking
                self.slo_tracker.record_trace(trace.to_dict())
                
            except Exception as e:
                logger.error(f"Error processing trace: {e}")
        
        self.collector.add_trace_listener(on_trace_complete)
    
    def receive_span(self, span_data: Dict[str, Any]) -> None:
        """Receive a span for processing"""
        span = Span.from_dict(span_data)
        self.collector.receive_span(span)
    
    def receive_trace(self, trace_data: Dict[str, Any]) -> None:
        """Receive a complete trace"""
        trace = Trace.from_dict(trace_data)
        self.collector.receive_trace(trace)
    
    def analyze_anomalies(
        self,
        time_window_minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        """Analyze recent traces for anomalies"""
        cutoff = datetime.utcnow().timestamp() - (time_window_minutes * 60)
        
        # Query recent traces
        filter = QueryFilter(limit=1000)
        traces = self.storage.query_traces(filter)
        
        # Filter by time
        recent_traces = []
        for trace in traces:
            if trace.get("start_time"):
                try:
                    ts = datetime.fromisoformat(trace["start_time"]).timestamp()
                    if ts >= cutoff:
                        recent_traces.append(trace)
                except (ValueError, TypeError):
                    continue
        
        if not self.anomaly_detector:
            return []
        
        # Detect anomalies
        scores = self.anomaly_detector.detect_batch(recent_traces)
        
        results = []
        for trace, score in zip(recent_traces, scores):
            if score.is_anomaly:
                results.append({
                    "trace_id": trace.get("trace_id"),
                    "anomaly_score": score.normalized,
                    "confidence": score.confidence,
                    "features": score.features,
                })
        
        return results
    
    def analyze_root_cause(
        self,
        time_window_minutes: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """Perform root cause analysis on recent traces"""
        cutoff = datetime.utcnow().timestamp() - (time_window_minutes * 60)
        
        # Query recent traces
        filter = QueryFilter(limit=100)
        traces = self.storage.query_traces(filter)
        
        # Filter by time
        recent_traces = []
        for trace in traces:
            if trace.get("start_time"):
                try:
                    ts = datetime.fromisoformat(trace["start_time"]).timestamp()
                    if ts >= cutoff:
                        recent_traces.append(trace)
                except (ValueError, TypeError):
                    continue
        
        if not recent_traces:
            return None
        
        # Analyze
        incident = self.root_cause_analyzer.analyze(recent_traces)
        
        return {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "description": incident.description,
            "severity": incident.severity.value,
            "root_cause_type": incident.root_cause_type.value,
            "affected_services": incident.affected_services,
            "contributing_factors": incident.contributing_factors,
            "recommendations": incident.recommendations,
            "start_time": incident.start_time.isoformat(),
        }
    
    def get_dependency_graph(
        self,
        time_window_minutes: int = 60,
    ) -> Dict[str, Any]:
        """Generate dependency graph from recent traces"""
        cutoff = datetime.utcnow().timestamp() - (time_window_minutes * 60)
        
        # Query recent traces
        filter = QueryFilter(limit=1000)
        traces = self.storage.query_traces(filter)
        
        # Filter by time
        recent_traces = []
        for trace in traces:
            if trace.get("start_time"):
                try:
                    ts = datetime.fromisoformat(trace["start_time"]).timestamp()
                    if ts >= cutoff:
                        recent_traces.append(trace)
                except (ValueError, TypeError):
                    continue
        
        # Generate graph
        graph = self.graph_generator.generate_from_traces(recent_traces)
        
        return {
            "graph": graph.to_dict(),
            "cytoscape": graph.to_cytoscape_json(),
            "health_summary": graph.get_health_summary(),
            "bottlenecks": self.graph_generator.find_bottlenecks(),
        }
    
    def get_slo_status(self) -> Dict[str, Any]:
        """Get current SLO status"""
        self.slo_tracker.evaluate_all_slos()
        
        return {
            "compliance": self.slo_tracker.get_compliance_report(),
            "alerts": self.slo_tracker.get_budget_burning_alerts(),
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        if self.alerter:
            return self.alerter.get_active_alerts()
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "uptime_seconds": (datetime.utcnow() - self.started_at).total_seconds(),
            "collector": self.collector.get_stats(),
            "storage": self.storage.get_stats(),
            "ml": self.anomaly_detector.get_stats() if self.anomaly_detector else {},
            "alerts": self.alerter.get_stats() if self.alerter else {},
        }
    
    def train_models(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train ML models on historical data"""
        if not self.anomaly_detector:
            return {"error": "ML not enabled"}
        
        logger.info(f"Training models on {len(traces)} traces...")
        
        metrics = self.anomaly_detector.train(traces, epochs=10)
        
        return {
            "traces_trained": len(traces),
            "metrics": {
                name: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                }
                for name, m in metrics.items()
            }
        }


def create_pipeline(
    service_name: str = "pulse",
    storage_backend: str = "memory",
) -> PulsePipeline:
    """Create a configured Pulse pipeline"""
    return PulsePipeline(
        service_name=service_name,
        storage_backend=storage_backend,
    )
