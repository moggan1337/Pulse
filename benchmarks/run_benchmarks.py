"""Benchmarks for Pulse"""

import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

import numpy as np


def generate_benchmark_traces(num_traces: int, spans_per_trace: int) -> List[Dict[str, Any]]:
    """Generate traces for benchmarking"""
    services = ["api-gateway", "user-service", "order-service", "payment-service", "inventory-service"]
    
    traces = []
    for i in range(num_traces):
        spans = []
        parent_id = None
        
        for j in range(spans_per_trace):
            service = random.choice(services)
            span_id = f"span-{i}-{j}"
            
            span = {
                "name": f"{service}.handler",
                "trace_id": f"trace-{i}",
                "span_id": span_id,
                "parent_span_id": parent_id,
                "service_name": service,
                "duration_ms": random.expovariate(1/100),
                "error_flag": random.random() < 0.05,
                "attributes": {
                    "http.status_code": 200 if not random.random() < 0.05 else 500,
                },
                "events": [],
            }
            
            spans.append(span)
            parent_id = span_id
        
        trace = {
            "trace_id": f"trace-{i}",
            "spans": spans,
            "start_time": datetime.utcnow().isoformat(),
            "total_duration_ms": sum(s["duration_ms"] for s in spans),
        }
        
        traces.append(trace)
    
    return traces


def benchmark_collector(num_spans: int = 10000):
    """Benchmark trace collector performance"""
    from src.collector import TraceCollector, Span, SpanKind
    
    collector = TraceCollector()
    
    services = ["api-gateway", "user-service", "order-service"]
    
    start = time.time()
    
    for i in range(num_spans):
        span = Span(
            name=f"span-{i}",
            trace_id=f"trace-{i // 10}",
            span_id=f"span-{i}",
            service_name=random.choice(services),
            kind=SpanKind.INTERNAL,
        )
        span.duration_ms = random.expovariate(1/100)
        span.finish()
        
        collector.receive_span(span)
    
    elapsed = time.time() - start
    
    print(f"\n=== Collector Benchmark ===")
    print(f"Spans processed: {num_spans}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {num_spans / elapsed:.0f} spans/sec")
    print(f"Collector stats: {collector.get_stats()}")


def benchmark_storage(num_traces: int = 1000, spans_per_trace: int = 10):
    """Benchmark storage performance"""
    from src.storage import TraceStorage, StorageConfig
    
    config = StorageConfig(backend="memory")
    storage = TraceStorage(config)
    
    traces = generate_benchmark_traces(num_traces, spans_per_trace)
    
    # Benchmark write
    start = time.time()
    for trace in traces:
        storage.store_trace_sync(trace)
    write_time = time.time() - start
    
    # Benchmark read
    start = time.time()
    for i in range(min(100, num_traces)):
        storage.get_trace(f"trace-{i}")
    read_time = time.time() - start
    
    # Benchmark query
    start = time.time()
    for _ in range(10):
        storage.query_traces(type('QueryFilter', (), {'limit': 100, 'offset': 0, 'service_name': None,
                                                       'error_only': False, 'trace_id': None,
                                                       'start_time': None, 'end_time': None,
                                                       'min_duration_ms': None, 'max_duration_ms': None})())
    query_time = time.time() - start
    
    print(f"\n=== Storage Benchmark ===")
    print(f"Traces: {num_traces}, Spans per trace: {spans_per_trace}")
    print(f"Write time: {write_time:.3f}s ({num_traces / write_time:.0f} traces/sec)")
    print(f"Read time (100 traces): {read_time:.3f}s")
    print(f"Query time (10 queries): {query_time:.3f}s")
    print(f"Storage stats: {storage.get_stats()}")


def benchmark_ml(num_traces: int = 500):
    """Benchmark ML anomaly detection"""
    try:
        from src.ml import create_detector, generate_synthetic_traces
        
        # Generate training data
        print(f"\nGenerating {num_traces} synthetic traces...")
        traces, _ = generate_synthetic_traces(num_traces, anomaly_rate=0.1)
        
        # Create detector
        detector = create_detector(enable_iforest=True, enable_lstm=False)
        detector.initialize()
        
        # Benchmark training
        print("Training model...")
        start = time.time()
        detector.train(traces[:int(num_traces * 0.8)])
        train_time = time.time() - start
        
        # Benchmark prediction
        print("Running predictions...")
        start = time.time()
        for trace in traces[int(num_traces * 0.8):]:
            detector.detect(trace)
        predict_time = time.time() - start
        
        predictions = num_traces - int(num_traces * 0.8)
        
        print(f"\n=== ML Benchmark ===")
        print(f"Training traces: {int(num_traces * 0.8)}")
        print(f"Train time: {train_time:.3f}s")
        print(f"Prediction time ({predictions} traces): {predict_time:.3f}s")
        print(f"Prediction throughput: {predictions / predict_time:.0f} traces/sec")
        print(f"Detector stats: {detector.get_stats()}")
        
    except ImportError as e:
        print(f"\n=== ML Benchmark (SKIPPED) ===")
        print(f"scikit-learn not available: {e}")


def benchmark_root_cause(num_traces: int = 100):
    """Benchmark root cause analysis"""
    from src.analysis import create_analyzer
    
    analyzer = create_analyzer()
    traces = generate_benchmark_traces(num_traces, spans_per_trace=5)
    
    start = time.time()
    incident = analyzer.analyze(traces)
    elapsed = time.time() - start
    
    print(f"\n=== Root Cause Analysis Benchmark ===")
    print(f"Traces analyzed: {num_traces}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {num_traces / elapsed:.0f} traces/sec")
    print(f"Incident ID: {incident.incident_id}")
    print(f"Root cause type: {incident.root_cause_type.value}")


def benchmark_graph_generation(num_traces: int = 500):
    """Benchmark dependency graph generation"""
    from src.graphs import create_graph_generator
    
    generator = create_graph_generator()
    traces = generate_benchmark_traces(num_traces, spans_per_trace=5)
    
    start = time.time()
    graph = generator.generate_from_traces(traces)
    elapsed = time.time() - start
    
    print(f"\n=== Graph Generation Benchmark ===")
    print(f"Traces: {num_traces}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print(f"Throughput: {num_traces / elapsed:.0f} traces/sec")


def benchmark_slo(num_traces: int = 1000):
    """Benchmark SLO tracking"""
    from src.slo import SLOTracker, SLOTarget, SLIType
    
    tracker = SLOTracker()
    
    slo = SLOTarget(
        slo_id="benchmark-availability",
        name="Benchmark Availability",
        description="Benchmark SLO",
        sli_type=SLIType.AVAILABILITY,
        target_value=0.999,
    )
    tracker.register_slo(slo)
    
    traces = generate_benchmark_traces(num_traces, spans_per_trace=3)
    
    # Benchmark recording
    start = time.time()
    for trace in traces:
        tracker.record_trace(trace)
    record_time = time.time() - start
    
    # Benchmark evaluation
    start = time.time()
    for _ in range(10):
        tracker.evaluate_slo("benchmark-availability")
    eval_time = time.time() - start
    
    print(f"\n=== SLO Tracking Benchmark ===")
    print(f"Traces recorded: {num_traces}")
    print(f"Record time: {record_time:.3f}s ({num_traces / record_time:.0f} traces/sec)")
    print(f"Evaluation time (10x): {eval_time:.3f}s")


def benchmark_context_propagation(num_operations: int = 100000):
    """Benchmark context propagation"""
    from src.context import TraceContext, ContextPropagator
    
    propagator = ContextPropagator()
    
    # Benchmark context creation
    start = time.time()
    contexts = []
    for i in range(num_operations):
        context = TraceContext.generate_new(service_name="test-service")
        contexts.append(context)
    create_time = time.time() - start
    
    # Benchmark injection
    start = time.time()
    for context in contexts[:10000]:
        carrier = {}
        propagator.inject_context(context, carrier)
    inject_time = time.time() - start
    
    # Benchmark extraction
    start = time.time()
    for context in contexts[:10000]:
        carrier = context.to_w3c_headers()
        extracted = TraceContext.from_w3c_headers(carrier)
    extract_time = time.time() - start
    
    print(f"\n=== Context Propagation Benchmark ===")
    print(f"Context creation ({num_operations}): {create_time:.3f}s ({num_operations / create_time:.0f} ops/sec)")
    print(f"Header injection (10000): {inject_time:.3f}s")
    print(f"Header extraction (10000): {extract_time:.3f}s")


def run_all_benchmarks():
    """Run all benchmarks"""
    print("=" * 60)
    print("PULSE BENCHMARK SUITE")
    print("=" * 60)
    
    # Collector
    benchmark_collector(num_spans=50000)
    
    # Storage
    benchmark_storage(num_traces=5000, spans_per_trace=5)
    
    # ML
    benchmark_ml(num_traces=300)
    
    # Root Cause Analysis
    benchmark_root_cause(num_traces=200)
    
    # Graph Generation
    benchmark_graph_generation(num_traces=1000)
    
    # SLO
    benchmark_slo(num_traces=2000)
    
    # Context Propagation
    benchmark_context_propagation(num_operations=50000)
    
    print("\n" + "=" * 60)
    print("BENCHMARKS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()
