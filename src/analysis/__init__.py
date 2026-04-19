"""Root Cause Analysis Module"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RootCauseType(Enum):
    """Types of root causes in distributed systems"""
    SERVICE_DEGRADATION = "service_degradation"
    DEPENDENCY_FAILURE = "dependency_failure"
    TIMEOUT = "timeout"
    ERROR_CASCADE = "error_cascade"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_CHANGE = "configuration_change"
    DEPLOYMENT_ISSUE = "deployment_issue"
    EXTERNAL_DEPENDENCY = "external_dependency"
    DATABASE_ISSUE = "database_issue"
    NETWORK_ISSUE = "network_issue"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Incident severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ServiceMetrics:
    """Aggregated metrics for a service"""
    service_name: str
    request_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    timeout_count: int = 0
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(1, self.request_count)
    
    @property
    def success_rate(self) -> float:
        return 1.0 - self.error_rate


@dataclass
class DependencyEdge:
    """Represents a dependency relationship between services"""
    source: str
    target: str
    call_count: int = 0
    error_count: int = 0
    avg_duration_ms: float = 0.0
    error_rate: float = 0.0
    
    @property
    def health_score(self) -> float:
        """Calculate health score (0-1)"""
        if self.call_count == 0:
            return 1.0
        return 1.0 - self.error_rate


@dataclass
class Incident:
    """Represents a detected incident"""
    incident_id: str
    title: str
    description: str
    severity: Severity
    root_cause_type: RootCauseType
    affected_services: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Causation chain
    trace_ids: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    
    # Metrics at incident time
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Status
    status: str = "open"
    assignee: Optional[str] = None
    
    def duration_seconds(self) -> float:
        """Calculate incident duration"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.utcnow() - self.start_time).total_seconds()


class TraceAnalyzer:
    """
    Analyzes traces to identify patterns and anomalies.
    
    Performs statistical analysis on trace data to identify:
    - Performance degradation
    - Error patterns
    - Latency outliers
    - Anomalous dependencies
    """
    
    def __init__(self):
        self.historical_baselines: Dict[str, Dict[str, float]] = {}
    
    def analyze_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a single trace.
        
        Returns analysis results including:
        - Anomaly indicators
        - Performance summary
        - Service dependencies
        - Error analysis
        """
        spans = trace.get("spans", [])
        if not spans:
            return {"error": "No spans in trace"}
        
        analysis = {
            "trace_id": trace.get("trace_id"),
            "total_spans": len(spans),
            "total_duration_ms": trace.get("total_duration_ms", 0),
            
            # Service analysis
            "services_involved": self._analyze_services(spans),
            
            # Performance analysis
            "performance": self._analyze_performance(spans),
            
            # Error analysis
            "errors": self._analyze_errors(spans),
            
            # Dependency analysis
            "dependencies": self._analyze_dependencies(spans),
            
            # Anomaly indicators
            "anomaly_indicators": self._detect_anomaly_indicators(spans),
        }
        
        return analysis
    
    def _analyze_services(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze service-level metrics"""
        service_spans = defaultdict(list)
        
        for span in spans:
            service = span.get("service_name", "unknown")
            service_spans[service].append(span)
        
        services = {}
        for service, service_span_list in service_spans.items():
            durations = [s.get("duration_ms", 0) for s in service_span_list]
            errors = sum(1 for s in service_span_list if s.get("error_flag"))
            
            services[service] = {
                "span_count": len(service_span_list),
                "avg_duration_ms": np.mean(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
                "error_count": errors,
                "error_rate": errors / len(service_span_list),
            }
        
        return services
    
    def _analyze_performance(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall performance characteristics"""
        durations = [s.get("duration_ms", 0) for s in spans]
        
        if not durations:
            return {}
        
        return {
            "total_duration_ms": sum(durations),
            "avg_span_duration_ms": np.mean(durations),
            "median_span_duration_ms": np.median(durations),
            "p95_span_duration_ms": np.percentile(durations, 95),
            "p99_span_duration_ms": np.percentile(durations, 99),
            "max_span_duration_ms": max(durations),
            "min_span_duration_ms": min(durations),
        }
    
    def _analyze_errors(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns"""
        error_spans = [s for s in spans if s.get("error_flag")]
        
        error_types = defaultdict(int)
        error_services = defaultdict(int)
        
        for span in error_spans:
            error_type = span.get("status_message") or span.get("attributes", {}).get("error.type", "unknown")
            error_types[error_type] += 1
            
            service = span.get("service_name", "unknown")
            error_services[service] += 1
        
        return {
            "total_errors": len(error_spans),
            "error_rate": len(error_spans) / len(spans) if spans else 0,
            "error_types": dict(error_types),
            "affected_services": list(error_services.keys()),
            "service_error_counts": dict(error_services),
        }
    
    def _analyze_dependencies(self, spans: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze service dependencies"""
        dependencies = defaultdict(set)
        
        for span in spans:
            current_service = span.get("service_name", "unknown")
            parent_span_id = span.get("parent_span_id")
            
            # Find parent service
            if parent_span_id:
                for parent_span in spans:
                    if parent_span.get("span_id") == parent_span_id:
                        parent_service = parent_span.get("service_name", "unknown")
                        if parent_service != current_service:
                            dependencies[current_service].add(parent_service)
                        break
        
        return {k: list(v) for k, v in dependencies.items()}
    
    def _detect_anomaly_indicators(self, spans: List[Dict[str, Any]]) -> List[str]:
        """Detect potential anomaly indicators"""
        indicators = []
        
        # Check for high error rate
        error_count = sum(1 for s in spans if s.get("error_flag"))
        if error_count / len(spans) > 0.1 if spans else False:
            indicators.append("high_error_rate")
        
        # Check for latency outliers
        durations = [s.get("duration_ms", 0) for s in spans]
        if durations:
            mean = np.mean(durations)
            std = np.std(durations)
            outliers = sum(1 for d in durations if abs(d - mean) > 3 * std)
            if outliers > len(spans) * 0.1:
                indicators.append("latency_outliers")
        
        # Check for timeouts
        timeout_spans = [s for s in spans if s.get("attributes", {}).get("timeout")]
        if timeout_spans:
            indicators.append("timeout_detected")
        
        # Check for deep call chains
        max_depth = self._calculate_max_depth(spans)
        if max_depth > 10:
            indicators.append("deep_call_chain")
        
        # Check for missing services
        service_span_counts = defaultdict(int)
        for span in spans:
            service_span_counts[span.get("service_name", "unknown")] += 1
        
        low_traffic_services = [s for s, count in service_span_counts.items() if count == 1]
        if low_traffic_services:
            indicators.append("low_traffic_services")
        
        return indicators
    
    def _calculate_max_depth(self, spans: List[Dict[str, Any]]) -> int:
        """Calculate maximum call chain depth"""
        span_map = {s.get("span_id"): s for s in spans}
        max_depth = 0
        
        for span in spans:
            depth = 0
            current = span
            while current.get("parent_span_id"):
                depth += 1
                parent = span_map.get(current["parent_span_id"])
                if not parent:
                    break
                current = parent
            max_depth = max(max_depth, depth)
        
        return max_depth


class RootCauseAnalyzer:
    """
    Performs root cause analysis on distributed traces.
    
    Uses multiple techniques:
    - Statistical analysis
    - Dependency graph analysis
    - Error propagation tracking
    - Time correlation
    """
    
    def __init__(self):
        self.trace_analyzer = TraceAnalyzer()
        self.service_graph: Dict[str, Dict[str, DependencyEdge]] = defaultdict(dict)
    
    def analyze(
        self,
        traces: List[Dict[str, Any]],
        incident_start: Optional[datetime] = None,
        incident_end: Optional[datetime] = None,
    ) -> Incident:
        """
        Perform root cause analysis on a set of traces.
        
        Identifies:
        - Root cause type
        - Affected services
        - Contributing factors
        - Recommendations
        """
        if not traces:
            return self._create_unknown_incident()
        
        # Build service dependency graph
        self._build_service_graph(traces)
        
        # Analyze each trace
        analyses = [self.trace_analyzer.analyze_trace(t) for t in traces]
        
        # Identify affected services
        affected_services = self._identify_affected_services(analyses)
        
        # Determine root cause type
        root_cause_type = self._determine_root_cause_type(analyses)
        
        # Identify contributing factors
        factors = self._identify_contributing_factors(analyses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            root_cause_type, 
            affected_services, 
            factors
        )
        
        # Create incident
        incident = Incident(
            incident_id=f"incident-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            title=self._generate_incident_title(root_cause_type, affected_services),
            description=self._generate_incident_description(analyses, factors),
            severity=self._determine_severity(analyses),
            root_cause_type=root_cause_type,
            affected_services=affected_services,
            start_time=incident_start or self._get_earliest_start(traces),
            end_time=incident_end,
            trace_ids=[t.get("trace_id") for t in traces],
            contributing_factors=factors,
            metrics_snapshot=self._snapshot_metrics(analyses),
            recommendations=recommendations,
        )
        
        return incident
    
    def _build_service_graph(self, traces: List[Dict[str, Any]]) -> None:
        """Build service dependency graph from traces"""
        self.service_graph.clear()
        
        for trace in traces:
            spans = trace.get("spans", [])
            span_map = {s.get("span_id"): s for s in spans}
            
            for span in spans:
                source_service = span.get("service_name", "unknown")
                
                if span.get("parent_span_id"):
                    parent = span_map.get(span["parent_span_id"])
                    if parent:
                        target_service = parent.get("service_name", "unknown")
                        
                        if source_service != target_service:
                            if target_service not in self.service_graph[source_service]:
                                self.service_graph[source_service][target_service] = DependencyEdge(
                                    source=source_service,
                                    target=target_service,
                                )
                            
                            edge = self.service_graph[source_service][target_service]
                            edge.call_count += 1
                            edge.avg_duration_ms = (
                                (edge.avg_duration_ms * (edge.call_count - 1) + span.get("duration_ms", 0)) 
                                / edge.call_count
                            )
                            if span.get("error_flag"):
                                edge.error_count += 1
                            edge.error_rate = edge.error_count / edge.call_count
    
    def _identify_affected_services(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify services affected by the incident"""
        service_impact = defaultdict(lambda: {"error_count": 0, "latency_impact": 0})
        
        for analysis in analyses:
            # Add errors
            errors = analysis.get("errors", {})
            for service, count in errors.get("service_error_counts", {}).items():
                service_impact[service]["error_count"] += count
            
            # Add latency impact
            perf = analysis.get("performance", {})
            for service, metrics in analysis.get("services_involved", {}).items():
                service_impact[service]["latency_impact"] += metrics.get("avg_duration_ms", 0)
        
        # Sort by impact
        affected = sorted(
            service_impact.keys(),
            key=lambda s: (
                service_impact[s]["error_count"] * 100 +
                service_impact[s]["latency_impact"] / 1000
            ),
            reverse=True,
        )
        
        return affected[:10]  # Top 10 affected services
    
    def _determine_root_cause_type(
        self, 
        analyses: List[Dict[str, Any]]
    ) -> RootCauseType:
        """Determine the most likely root cause type"""
        indicators = defaultdict(int)
        
        for analysis in analyses:
            for indicator in analysis.get("anomaly_indicators", []):
                indicators[indicator] += 1
        
        # Map indicators to root cause types
        if indicators.get("timeout_detected"):
            return RootCauseType.TIMEOUT
        
        if indicators.get("high_error_rate"):
            return RootCauseType.ERROR_CASCADE
        
        if indicators.get("latency_outliers"):
            return RootCauseType.SERVICE_DEGRADATION
        
        # Analyze error types
        error_types = defaultdict(int)
        for analysis in analyses:
            for error_type, count in analysis.get("errors", {}).get("error_types", {}).items():
                error_types[error_type] += count
        
        if "database" in str(error_types).lower() or "connection" in str(error_types).lower():
            return RootCauseType.DATABASE_ISSUE
        
        if "network" in str(error_types).lower() or "timeout" in str(error_types).lower():
            return RootCauseType.NETWORK_ISSUE
        
        return RootCauseType.UNKNOWN
    
    def _identify_contributing_factors(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify factors contributing to the incident"""
        factors = []
        
        # Aggregate all indicators
        all_indicators = []
        for analysis in analyses:
            all_indicators.extend(analysis.get("anomaly_indicators", []))
        
        indicator_counts = defaultdict(int)
        for indicator in all_indicators:
            indicator_counts[indicator] += 1
        
        # Add frequent indicators as factors
        for indicator, count in indicator_counts.items():
            if count >= len(analyses) * 0.3:  # Appears in 30%+ of traces
                factors.append(f"{indicator}: affected {count} traces")
        
        # Add service-specific factors
        service_errors = defaultdict(int)
        for analysis in analyses:
            for service, error_count in analysis.get("errors", {}).get("service_error_counts", {}).items():
                service_errors[service] += error_count
        
        for service, errors in sorted(service_errors.items(), key=lambda x: x[1], reverse=True)[:3]:
            if errors > 0:
                factors.append(f"{service}: {errors} errors")
        
        return factors
    
    def _generate_recommendations(
        self,
        root_cause_type: RootCauseType,
        affected_services: List[str],
        factors: List[str],
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        type_recommendations = {
            RootCauseType.SERVICE_DEGRADATION: [
                "Check service resource utilization (CPU, memory)",
                "Review recent deployments to this service",
                "Examine service logs for errors or warnings",
                "Consider scaling horizontally or optimizing queries",
            ],
            RootCauseType.DEPENDENCY_FAILURE: [
                "Check health of downstream dependencies",
                "Review dependency timeout configurations",
                "Implement circuit breaker pattern",
                "Add retry logic with exponential backoff",
            ],
            RootCauseType.TIMEOUT: [
                "Increase timeout thresholds if appropriate",
                "Optimize slow database queries",
                "Implement async processing for long operations",
                "Add caching to reduce response times",
            ],
            RootCauseType.ERROR_CASCADE: [
                "Implement bulkhead pattern to isolate failures",
                "Add graceful degradation for non-critical features",
                "Review error handling in calling services",
                "Implement proper error propagation",
            ],
            RootCauseType.DATABASE_ISSUE: [
                "Check database connection pool settings",
                "Review slow query logs",
                "Consider adding database indexes",
                "Check for long-running transactions",
            ],
            RootCauseType.NETWORK_ISSUE: [
                "Check network latency between services",
                "Review load balancer configuration",
                "Verify DNS resolution",
                "Check for network partition events",
            ],
        }
        
        recommendations.extend(type_recommendations.get(root_cause_type, [
            "Investigate service logs",
            "Review recent changes",
            "Check monitoring dashboards",
        ]))
        
        # Add service-specific recommendations
        if affected_services:
            recommendations.append(
                f"Focus investigation on: {', '.join(affected_services[:3])}"
            )
        
        return recommendations
    
    def _determine_severity(self, analyses: List[Dict[str, Any]]) -> Severity:
        """Determine incident severity"""
        total_traces = len(analyses)
        if total_traces == 0:
            return Severity.INFO
        
        # Calculate aggregate metrics
        total_errors = sum(a.get("errors", {}).get("total_errors", 0) for a in analyses)
        error_rate = total_errors / sum(a.get("total_spans", 1) for a in analyses)
        
        # Calculate latency impact
        p95_latencies = [a.get("performance", {}).get("p95_span_duration_ms", 0) for a in analyses]
        avg_p95 = np.mean(p95_latencies) if p95_latencies else 0
        
        # Determine severity
        if error_rate > 0.5 or avg_p95 > 10000:
            return Severity.CRITICAL
        elif error_rate > 0.2 or avg_p95 > 5000:
            return Severity.HIGH
        elif error_rate > 0.05 or avg_p95 > 2000:
            return Severity.MEDIUM
        elif error_rate > 0 or avg_p95 > 1000:
            return Severity.LOW
        else:
            return Severity.INFO
    
    def _generate_incident_title(
        self,
        root_cause_type: RootCauseType,
        affected_services: List[str],
    ) -> str:
        """Generate a human-readable incident title"""
        service_str = ", ".join(affected_services[:2]) if affected_services else "Multiple services"
        
        type_titles = {
            RootCauseType.SERVICE_DEGRADATION: f"Performance degradation in {service_str}",
            RootCauseType.DEPENDENCY_FAILURE: f"Dependency failure affecting {service_str}",
            RootCauseType.TIMEOUT: f"Timeout errors in {service_str}",
            RootCauseType.ERROR_CASCADE: f"Error cascade in {service_str}",
            RootCauseType.DATABASE_ISSUE: f"Database issues affecting {service_str}",
            RootCauseType.NETWORK_ISSUE: f"Network issues affecting {service_str}",
            RootCauseType.UNKNOWN: f"Unknown issue affecting {service_str}",
        }
        
        return type_titles.get(root_cause_type, f"Incident in {service_str}")
    
    def _generate_incident_description(
        self,
        analyses: List[Dict[str, Any]],
        factors: List[str],
    ) -> str:
        """Generate incident description"""
        total_errors = sum(a.get("errors", {}).get("total_errors", 0) for a in analyses)
        total_spans = sum(a.get("total_spans", 0) for a in analyses)
        
        p95_latencies = [a.get("performance", {}).get("p95_span_duration_ms", 0) for a in analyses]
        max_p95 = max(p95_latencies) if p95_latencies else 0
        
        description = f"""
Analysis of {len(analyses)} traces revealed:
- Total errors: {total_errors} ({total_errors/max(total_spans, 1)*100:.1f}% error rate)
- Peak P95 latency: {max_p95:.2f}ms
- Contributing factors: {len(factors)}

{' '.join(factors[:3])}
        """.strip()
        
        return description
    
    def _get_earliest_start(self, traces: List[Dict[str, Any]]) -> datetime:
        """Get the earliest start time from traces"""
        start_times = []
        for trace in traces:
            start_time_str = trace.get("start_time")
            if start_time_str:
                try:
                    start_times.append(datetime.fromisoformat(start_time_str))
                except ValueError:
                    pass
        
        return min(start_times) if start_times else datetime.utcnow()
    
    def _snapshot_metrics(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a snapshot of metrics at incident time"""
        return {
            "total_traces_analyzed": len(analyses),
            "total_errors": sum(a.get("errors", {}).get("total_errors", 0) for a in analyses),
            "avg_p95_latency": np.mean([a.get("performance", {}).get("p95_span_duration_ms", 0) for a in analyses]),
            "services_affected": len(set().union(*[set(a.get("services_involved", {}).keys()) for a in analyses])),
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def _create_unknown_incident(self) -> Incident:
        """Create an unknown incident when no data is available"""
        return Incident(
            incident_id=f"incident-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            title="Unknown incident",
            description="No trace data available for analysis",
            severity=Severity.INFO,
            root_cause_type=RootCauseType.UNKNOWN,
            affected_services=[],
            start_time=datetime.utcnow(),
        )
    
    def get_service_health_scores(self) -> Dict[str, float]:
        """Calculate health scores for all services based on dependency graph"""
        scores = {}
        
        for service, dependencies in self.service_graph.items():
            if not dependencies:
                scores[service] = 1.0
            else:
                # Health based on dependency health
                dep_health = [d.health_score for d in dependencies.values()]
                scores[service] = np.mean(dep_health)
        
        return scores
    
    def find_weakest_link(self) -> Optional[Tuple[str, str]]:
        """Find the dependency edge with the lowest health score"""
        weakest = None
        min_health = 1.0
        
        for service, dependencies in self.service_graph.items():
            for target, edge in dependencies.items():
                if edge.health_score < min_health:
                    min_health = edge.health_score
                    weakest = (service, target)
        
        return weakest


class CausalAnalyzer:
    """
    Analyzes causal relationships between services using trace data.
    
    Builds a causal graph showing how errors and latency propagate
    through the system.
    """
    
    def __init__(self):
        self.causal_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def analyze_causality(
        self,
        traces: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze causal relationships from trace data.
        
        Returns:
            Causal graph and analysis results
        """
        self.causal_graph.clear()
        
        for trace in traces:
            self._analyze_trace_causality(trace)
        
        return {
            "causal_graph": {k: dict(v) for k, v in self.causal_graph.items()},
            "root_causes": self._identify_root_causes(),
            "affected_services": self._get_causally_affected_services(),
        }
    
    def _analyze_trace_causality(self, trace: Dict[str, Any]) -> None:
        """Analyze causality within a single trace"""
        spans = trace.get("spans", [])
        span_map = {s.get("span_id"): s for s in spans}
        
        # Find error chain
        error_spans = [s for s in spans if s.get("error_flag")]
        
        for error_span in error_spans:
            source_service = error_span.get("service_name", "unknown")
            
            # Find what caused this error (parent)
            if error_span.get("parent_span_id"):
                parent = span_map.get(error_span["parent_span_id"])
                if parent:
                    target_service = parent.get("service_name", "unknown")
                    self.causal_graph[source_service][target_service] += 1
    
    def _identify_root_causes(self) -> List[str]:
        """Identify likely root cause services"""
        # A root cause is a service that causes errors but isn't caused by others
        causes = defaultdict(int)
        effects = defaultdict(int)
        
        for source, targets in self.causal_graph.items():
            causes[source] += sum(targets.values())
            for target in targets:
                effects[target] += targets[target]
        
        root_causes = []
        for service in causes:
            if causes[service] > effects[service]:
                root_causes.append(service)
        
        return sorted(root_causes, key=lambda s: causes[s], reverse=True)
    
    def _get_causally_affected_services(self) -> List[str]:
        """Get services affected by root causes"""
        affected = set()
        
        for source, targets in self.causal_graph.items():
            for target in targets:
                affected.add(target)
        
        return list(affected)


def create_analyzer() -> RootCauseAnalyzer:
    """Create a configured root cause analyzer"""
    return RootCauseAnalyzer()
