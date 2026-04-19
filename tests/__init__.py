"""Tests for Pulse"""

import pytest
from datetime import datetime, timedelta
from src.collector import TraceCollector, Trace, Span, SpanKind
from src.storage import TraceStorage, StorageConfig, QueryFilter
from src.ml import AnomalyDetector, create_detector, generate_synthetic_traces
from src.analysis import RootCauseAnalyzer, create_analyzer
from src.alerting import PredictiveAlerter, create_alerter, AlertRule, AlertSeverity
from src.graphs import DependencyGraphGenerator, create_graph_generator
from src.slo import SLOTracker, SLOTarget, SLIType
from src.context import TraceContext, ContextPropagator


class TestCollector:
    """Tests for TraceCollector"""
    
    def test_create_collector(self):
        collector = TraceCollector(service_name="test")
        assert collector.service_name == "test"
        assert collector.stats["spans_received"] == 0
    
    def test_receive_span(self):
        collector = TraceCollector()
        span = Span(
            name="test-span",
            trace_id="trace-123",
            span_id="span-456",
            service_name="test-service",
        )
        collector.receive_span(span)
        assert collector.stats["spans_received"] == 1
    
    def test_trace_completion(self):
        collector = TraceCollector()
        
        # Create spans
        parent = Span(
            name="parent",
            trace_id="trace-123",
            span_id="parent-span",
            service_name="service-a",
        )
        parent.finish()
        
        child = Span(
            name="child",
            trace_id="trace-123",
            span_id="child-span",
            parent_span_id="parent-span",
            service_name="service-b",
        )
        child.finish()
        
        collector.receive_span(parent)
        collector.receive_span(child)
        
        # Check trace is stored
        trace = collector.get_trace("trace-123")
        assert trace is not None
        assert len(trace.spans) == 2


class TestStorage:
    """Tests for TraceStorage"""
    
    def test_memory_storage(self):
        config = StorageConfig(backend="memory")
        storage = TraceStorage(config)
        
        # Store a trace
        trace_data = {
            "trace_id": "test-123",
            "spans": [
                {
                    "name": "span1",
                    "trace_id": "test-123",
                    "span_id": "span-1",
                    "service_name": "service-a",
                    "duration_ms": 100,
                    "error_flag": False,
                }
            ],
            "start_time": datetime.utcnow().isoformat(),
            "total_duration_ms": 100,
        }
        
        # Synchronous storage
        storage.store_trace_sync(trace_data)
        
        # Retrieve
        retrieved = storage.get_trace("test-123")
        assert retrieved is not None
        assert retrieved["trace_id"] == "test-123"
    
    def test_query_filter(self):
        config = StorageConfig(backend="memory")
        storage = TraceStorage(config)
        
        # Store multiple traces
        for i in range(5):
            trace_data = {
                "trace_id": f"trace-{i}",
                "spans": [
                    {
                        "name": "span",
                        "trace_id": f"trace-{i}",
                        "span_id": f"span-{i}",
                        "service_name": "service-a",
                        "duration_ms": 100 + i * 10,
                        "error_flag": i % 2 == 0,
                    }
                ],
                "start_time": datetime.utcnow().isoformat(),
                "total_duration_ms": 100 + i * 10,
            }
            storage.store_trace_sync(trace_data)
        
        # Query with filter
        filter = QueryFilter(service_name="service-a", limit=10)
        results = storage.query_traces(filter)
        assert len(results) == 5


class TestML:
    """Tests for ML Anomaly Detection"""
    
    def test_generate_synthetic_data(self):
        traces, labels = generate_synthetic_traces(num_traces=10, anomaly_rate=0.3)
        assert len(traces) == 10
        assert len(labels) == 10
        assert sum(labels) > 0  # Should have some anomalies
    
    def test_detector_creation(self):
        try:
            detector = create_detector(enable_iforest=True)
            assert detector is not None
        except ImportError:
            pytest.skip("scikit-learn not installed")


class TestRootCause:
    """Tests for Root Cause Analysis"""
    
    def test_create_analyzer(self):
        analyzer = create_analyzer()
        assert analyzer is not None
    
    def test_analyze_traces(self):
        analyzer = create_analyzer()
        
        # Create sample traces
        traces = []
        for i in range(5):
            trace = {
                "trace_id": f"trace-{i}",
                "spans": [
                    {
                        "name": "span1",
                        "trace_id": f"trace-{i}",
                        "span_id": f"span-{i}-1",
                        "service_name": "api-gateway",
                        "duration_ms": 100,
                        "error_flag": i == 0,  # First trace has error
                        "attributes": {},
                        "events": [],
                    },
                    {
                        "name": "span2",
                        "trace_id": f"trace-{i}",
                        "span_id": f"span-{i}-2",
                        "parent_span_id": f"span-{i}-1",
                        "service_name": "user-service",
                        "duration_ms": 50,
                        "error_flag": False,
                        "attributes": {},
                        "events": [],
                    },
                ],
                "start_time": datetime.utcnow().isoformat(),
                "total_duration_ms": 150,
            }
            traces.append(trace)
        
        incident = analyzer.analyze(traces)
        
        assert incident is not None
        assert incident.incident_id is not None
        assert len(incident.affected_services) > 0


class TestAlerting:
    """Tests for Predictive Alerting"""
    
    def test_create_alerter(self):
        alerter = create_alerter()
        assert alerter is not None
    
    def test_add_rule(self):
        alerter = create_alerter()
        
        rule = AlertRule(
            rule_id="test-rule",
            name="Test Alert",
            description="A test alert rule",
            alert_type=alerter.threshold_detector.rules.get("error-rate-high").alert_type if alerter.threshold_detector.rules else None,
            severity=AlertSeverity.HIGH,
            metric_name="error_rate",
            condition="gt",
            threshold=0.1,
        )
        
        # Check alerter has default rules
        stats = alerter.get_stats()
        assert "total_alerts_fired" in stats
    
    def test_update_metrics(self):
        alerter = create_alerter()
        
        metrics = {
            "api-gateway": {
                "error_rate": 0.02,
                "p95_duration_ms": 500,
                "success_rate": 0.98,
            }
        }
        
        alerter.update_metrics(metrics)
        
        # Check metrics are tracked
        summary = alerter.get_alert_summary()
        assert summary is not None


class TestGraphs:
    """Tests for Dependency Graphs"""
    
    def test_create_generator(self):
        generator = create_graph_generator()
        assert generator is not None
    
    def test_generate_from_traces(self):
        generator = create_graph_generator()
        
        traces = [
            {
                "trace_id": "trace-1",
                "spans": [
                    {
                        "name": "api-handler",
                        "trace_id": "trace-1",
                        "span_id": "span-1",
                        "service_name": "api-gateway",
                        "duration_ms": 100,
                        "error_flag": False,
                    },
                    {
                        "name": "user-handler",
                        "trace_id": "trace-1",
                        "span_id": "span-2",
                        "parent_span_id": "span-1",
                        "service_name": "user-service",
                        "duration_ms": 50,
                        "error_flag": False,
                    },
                    {
                        "name": "db-query",
                        "trace_id": "trace-1",
                        "span_id": "span-3",
                        "parent_span_id": "span-2",
                        "service_name": "database",
                        "duration_ms": 30,
                        "error_flag": False,
                    },
                ],
                "start_time": datetime.utcnow().isoformat(),
                "total_duration_ms": 180,
            }
        ]
        
        graph = generator.generate_from_traces(traces)
        
        assert len(graph.nodes) == 3
        assert len(graph.edges) >= 2
        
        # Check nodes exist
        assert "api-gateway" in [n.name for n in graph.nodes.values()]
        assert "user-service" in [n.name for n in graph.nodes.values()]
    
    def test_health_summary(self):
        generator = create_graph_generator()
        
        traces = [
            {
                "trace_id": "trace-1",
                "spans": [
                    {
                        "name": "span1",
                        "trace_id": "trace-1",
                        "span_id": "span-1",
                        "service_name": "service-a",
                        "duration_ms": 100,
                        "error_flag": False,
                    },
                    {
                        "name": "span2",
                        "trace_id": "trace-1",
                        "span_id": "span-2",
                        "parent_span_id": "span-1",
                        "service_name": "service-b",
                        "duration_ms": 50,
                        "error_flag": True,  # Error
                    },
                ],
                "start_time": datetime.utcnow().isoformat(),
                "total_duration_ms": 150,
            }
        ]
        
        graph = generator.generate_from_traces(traces)
        summary = graph.get_health_summary()
        
        assert "overall_health" in summary
        assert summary["total_nodes"] == 2


class TestSLO:
    """Tests for SLO Tracking"""
    
    def test_create_tracker(self):
        tracker = SLOTracker()
        assert tracker is not None
    
    def test_register_slo(self):
        tracker = SLOTracker()
        
        slo = SLOTarget(
            slo_id="test-availability",
            name="Test Availability",
            description="Test SLO",
            sli_type=SLIType.AVAILABILITY,
            target_value=0.999,
        )
        
        tracker.register_slo(slo)
        
        assert "test-availability" in tracker.slos
    
    def test_record_and_evaluate(self):
        tracker = SLOTracker()
        
        slo = SLOTarget(
            slo_id="test-availability",
            name="Test Availability",
            description="Test SLO",
            sli_type=SLIType.AVAILABILITY,
            target_value=0.999,
        )
        
        tracker.register_slo(slo)
        
        # Record traces
        for i in range(100):
            trace = {
                "trace_id": f"trace-{i}",
                "spans": [
                    {
                        "name": "span",
                        "trace_id": f"trace-{i}",
                        "span_id": f"span-{i}",
                        "service_name": "test-service",
                        "duration_ms": 100,
                        "error_flag": i < 99,  # 1% error rate
                    }
                ],
                "start_time": datetime.utcnow().isoformat(),
                "total_duration_ms": 100,
            }
            tracker.record_trace(trace)
        
        # Evaluate
        compliance = tracker.evaluate_slo("test-availability")
        
        assert compliance is not None
        assert compliance.current_value > 0.98


class TestContext:
    """Tests for Context Propagation"""
    
    def test_create_context(self):
        context = TraceContext.generate_new(service_name="test-service")
        assert context.trace_id is not None
        assert context.span_id is not None
    
    def test_traceparent_format(self):
        context = TraceContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags="01",
        )
        
        traceparent = context.to_traceparent()
        assert traceparent == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    
    def test_parse_traceparent(self):
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        
        context = TraceContext.from_traceparent(traceparent)
        
        assert context is not None
        assert context.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert context.span_id == "b7ad6b7169203331"
        assert context.is_sampled is True
    
    def test_child_context(self):
        parent = TraceContext.generate_new(service_name="parent-service")
        
        child = parent.create_child_context(
            span_id="child-span-id",
            operation_name="child-operation",
        )
        
        assert child.trace_id == parent.trace_id
        assert child.span_id == "child-span-id"
        assert child.operation_name == "child-operation"
    
    def test_baggage(self):
        context = TraceContext.generate_new()
        
        context.set_baggage("user-id", "12345")
        context.set_baggage("tenant-id", "acme")
        
        assert context.get_baggage("user-id") == "12345"
        assert context.get_baggage("tenant-id") == "acme"
        assert context.get_baggage("nonexistent") is None
    
    def test_propagator(self):
        propagator = ContextPropagator()
        
        context = TraceContext.generate_new(service_name="test-service")
        
        carrier = {}
        propagator.inject_context(context, carrier)
        
        assert "traceparent" in carrier
        
        # Extract
        extracted = propagator.extract_context(carrier)
        assert extracted is not None
        assert extracted.trace_id == context.trace_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
