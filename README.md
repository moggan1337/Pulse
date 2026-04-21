# Pulse - Distributed Tracing with ML Anomaly Detection

<p align="center">
  <img src="docs/logo.png" alt="Pulse Logo" width="200"/>
</p>

<p align="center">
  <strong>Intelligent Observability for Distributed Systems</strong>
</p>

<p align="center">
  <a href="https://github.com/moggan1337/Pulse/actions">
    <img src="https://github.com/moggan1337/Pulse/workflows/CI/badge.svg" alt="CI"/>
  </a>
  <a href="https://pypi.org/project/pulse-observability/">
    <img src="https://img.shields.io/pypi/v/pulse-observability.svg" alt="PyPI"/>
  </a>
  <a href="https://pulse.readthedocs.io/">
    <img src="https://readthedocs.org/projects/pulse/badge/?version=latest" alt="Documentation"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
  </a>
</p>

---

## 🎬 Demo
![Pulse Demo](demo.gif)

*Distributed tracing with ML anomaly detection*

## Screenshots
| Component | Preview |
|-----------|---------|
| Trace View | ![trace](screenshots/trace-view.png) |
| Anomaly Detection | ![anomaly](screenshots/anomaly.png) |
| Service Map | ![map](screenshots/service-map.png) |

## Visual Description
Trace view shows request flow through services with timing. Anomaly detection highlights anomalous spans with severity. Service map displays dependency graph with traffic patterns.

---


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [ML Models](#ml-models)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Benchmarks](#benchmarks)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Pulse is a next-generation distributed tracing platform that combines traditional observability with machine learning-powered anomaly detection. Built on OpenTelemetry standards, Pulse provides intelligent insights into complex microservice architectures, automatically identifying performance issues, predicting failures, and accelerating root cause analysis.

### Why Pulse?

Traditional monitoring tells you **what** happened. Pulse tells you **why** it happened and **what** will happen next.

- **Proactive Detection**: ML models detect anomalies before they become outages
- **Automated Root Cause**: AI-powered analysis pinpoints issues in seconds
- **Predictive Alerting**: Forecast problems hours before they occur
- **Visual Understanding**: Interactive dependency graphs and service maps

---

## Features

### Core Tracing Capabilities

| Feature | Description |
|---------|-------------|
| **OpenTelemetry Native** | Full compatibility with OpenTelemetry SDKs and collectors |
| **Distributed Context Propagation** | W3C Trace Context and B3 propagation support |
| **Multi-Format Storage** | In-memory, SQLite, Redis, and Elasticsearch backends |
| **High-Performance Ingestion** | Handles 100K+ spans per second |
| **Flexible Sampling** | Configurable per-service sampling strategies |

### ML-Powered Intelligence

| Feature | Description |
|---------|-------------|
| **Isolation Forest** | Unsupervised anomaly detection using tree-based algorithms |
| **LSTM Networks** | Time-series forecasting for predictive alerting |
| **Ensemble Detection** | Combined models for higher accuracy |
| **Online Learning** | Continuous model improvement from new data |
| **Confidence Scoring** | Understand how certain the model is about predictions |

### Analysis & Visualization

| Feature | Description |
|---------|-------------|
| **Root Cause Analysis** | Automatic causal chain identification |
| **Dependency Graphs** | Service topology visualization |
| **Bottleneck Detection** | Identify slow dependencies |
| **SLO Tracking** | Service Level Objective monitoring |
| **Error Cascade Analysis** | Track how errors propagate |

### Alerting & Actions

| Feature | Description |
|---------|-------------|
| **Predictive Alerts** | Forecast issues before they occur |
| **Threshold Alerts** | Traditional metric-based alerting |
| **SLO Budget Burn** | Monitor error budget consumption |
| **Custom Rules** | Define custom alerting conditions |
| **Multi-Channel** | Webhook, Slack, PagerDuty support |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PULSE ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

                                ┌─────────────────┐
                                │   Applications  │
                                │  (Microservices)│
                                └────────┬────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │ OpenTelemetry SDK  │                    │
                    │  - Auto-instrumentation                 │
                    │  - Manual spans                         │
                    │  - Context propagation                   │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION LAYER                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  HTTP Endpoint   │  │  gRPC (OTLP)     │  │  Kafka/MQ        │         │
│  │  /api/v1/traces  │  │  :4317, :4318    │  │  Consumer        │         │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘         │
│           │                      │                      │                   │
│           └──────────────────────┼──────────────────────┘                   │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      SPAN PROCESSOR                                  │  │
│  │  - Sampling decisions                                               │  │
│  │  - Attribute enrichment                                            │  │
│  │  - Trace assembly                                                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROCESSING PIPELINE                               │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   STORAGE   │    │      ML      │    │   ANALYSIS  │    │  ALERTING   │  │
│  │             │    │             │    │             │    │             │  │
│  │ - Traces    │    │ - Isolation │    │ - Root      │    │ - Threshold │  │
│  │ - Indexes   │    │   Forest    │    │   Cause     │    │ - Predictive│  │
│  │ - Metrics   │    │ - LSTM      │    │ - Graphs    │    │ - SLO Burn  │  │
│  │             │    │ - Ensemble  │    │ - Bottleneck│    │             │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │          │
│         └──────────────────┴──────────────────┴──────────────────┘          │
│                                    │                                         │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   REST API   │  │  WebSocket   │  │   GraphQL   │  │   Metrics    │     │
│  │  /api/v1/*   │  │  Streaming  │  │   (future)  │  │  /metrics    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         VISUALIZATION                                 │  │
│  │  - Trace Explorer     - Dependency Graph    - SLO Dashboard          │  │
│  │  - Anomaly Timeline   - Service Map         - Alert Management       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### 1. Trace Collector (`src/collector/`)

The collector is responsible for receiving and processing spans from instrumented services.

**Key Classes:**
- `TraceCollector`: Main collector class with OpenTelemetry integration
- `Span`: Represents a single span with timing and attributes
- `Trace`: Collection of related spans forming a distributed trace
- `SpanProcessor`: Pipeline for span transformation and sampling

**Features:**
- OpenTelemetry SDK integration
- Custom sampling strategies
- Attribute enrichment
- Real-time trace assembly
- Event and link tracking

#### 2. Storage Layer (`src/storage/`)

Multiple storage backends for different scale requirements.

**Backends:**
- **Memory**: Fast, ephemeral storage for development/testing
- **SQLite**: Persistent storage for single-node deployments
- **Redis**: Distributed caching with TTL support
- **Elasticsearch**: Full-text search and analytics

**Key Classes:**
- `TraceStorage`: Unified storage interface
- `StorageBackend`: Abstract base for storage implementations
- `TraceAggregator`: Aggregation for analytics

#### 3. ML Anomaly Detection (`src/ml/`)

Machine learning models for intelligent anomaly detection.

**Models:**

##### Isolation Forest
- Unsupervised learning algorithm
- Detects outliers in high-dimensional spaces
- No labeled data required
- Fast training and inference

**Algorithm Details:**
```
Isolation Forest Algorithm:
1. Build random trees by selecting random features and split values
2. Anomalies are isolated closer to the root (shorter path length)
3. Calculate average path length for each point
4. Anomaly score = 2^(-avg_path_length / c(n))
5. Lower score = more anomalous
```

##### LSTM Neural Network
- Recurrent neural network for sequence modeling
- Learns temporal patterns in trace data
- Predicts expected behavior based on history
- Flags deviations from predicted patterns

**Architecture:**
```
LSTM Anomaly Detection:
Input: Time-series features (latency, errors, throughput)
      ↓
LSTM Layer 1 (64 units)
      ↓
LSTM Layer 2 (64 units)
      ↓
Dense Layer
      ↓
Output: Reconstruction error (high = anomalous)
```

##### Ensemble Detection
- Combines multiple models
- Weighted voting based on confidence
- More robust than single models
- Handles diverse anomaly types

**Feature Engineering:**
```python
Features extracted per trace:
- Duration statistics (mean, std, min, max, p50, p95)
- Span count and composition
- Error rate and types
- Service diversity
- Service-level duration metrics
- Event count
- Link count (distributed context)
- Parent-child relationships
- Historical deviation scores
```

#### 4. Root Cause Analysis (`src/analysis/`)

Automated root cause identification using trace analysis.

**Analysis Techniques:**
- **Statistical Analysis**: Identify outliers and patterns
- **Dependency Graph Analysis**: Map service relationships
- **Error Propagation**: Track error cascades
- **Time Correlation**: Correlate metrics across services

**Key Classes:**
- `TraceAnalyzer`: Individual trace analysis
- `RootCauseAnalyzer`: Multi-trace analysis
- `CausalAnalyzer`: Causal relationship detection

**Output:**
```json
{
  "incident_id": "incident-20240115-143022",
  "title": "Performance degradation in api-gateway, user-service",
  "severity": "high",
  "root_cause_type": "service_degradation",
  "affected_services": ["api-gateway", "user-service", "order-service"],
  "contributing_factors": [
    "high_error_rate: affected 15 traces",
    "latency_outliers: affected 23 traces"
  ],
  "recommendations": [
    "Check service resource utilization (CPU, memory)",
    "Review recent deployments to this service"
  ]
}
```

#### 5. Predictive Alerting (`src/alerting/`)

Intelligent alerting system with forecasting capabilities.

**Alert Types:**
- **Threshold Alerts**: Traditional metric-based alerts
- **Anomaly Alerts**: ML-detected anomalies
- **Predictive Alerts**: Forecasted future issues
- **SLO Alerts**: Error budget and availability breaches

**Prediction Models:**
- Moving Average
- Exponential Smoothing
- Trend Extrapolation
- Anomaly-based Forecasting

#### 6. Dependency Graphs (`src/graphs/`)

Service topology visualization and analysis.

**Graph Features:**
- Node types: Service, Database, Cache, Queue, External
- Edge types: HTTP, gRPC, Database, Cache, Message Queue
- Health scores per node and edge
- Critical path identification
- Bottleneck detection

**Visualization Format:**
Cytoscape.js compatible JSON for integration with graph visualization tools.

#### 7. SLO Tracking (`src/slo/`)

Service Level Objectives monitoring and reporting.

**Supported SLIs:**
- Availability (success rate)
- Latency (P50, P95, P99)
- Correctness (error rate)
- Throughput (requests per second)

**Error Budget:**
```
Error Budget Calculation:
Target: 99.9% availability over 30 days
Allowed downtime: 43.8 minutes per month
Current: 99.85% availability
Budget remaining: 81.2%
Burn rate: 1.3x (burning faster than expected)
Time to exhaustion: 15.6 hours
```

#### 8. Context Propagation (`src/context/`)

Distributed tracing context management.

**Supported Formats:**
- W3C Trace Context (recommended)
- B3 (Zipkin style)
- Custom formats

**Features:**
- Context injection/extraction
- Baggage propagation
- Multi-format support
- HTTP, gRPC, and message queue adapters

---

## Quick Start

### Docker Compose

```bash
# Clone the repository
git clone https://github.com/moggan1337/Pulse.git
cd Pulse

# Start all services
docker-compose up -d

# Access the dashboard
open http://localhost:8080
```

### Python Installation

```bash
# Install Pulse
pip install pulse-observability

# Or install from source
git clone https://github.com/moggan1337/Pulse.git
cd Pulse
pip install -e .
```

### Basic Usage

```python
from pulse import PulsePipeline

# Create pipeline
pulse = PulsePipeline(
    service_name="my-service",
    storage_backend="memory",
    enable_ml=True,
    enable_alerting=True,
)

# Receive traces
pulse.receive_trace({
    "trace_id": "abc123",
    "spans": [
        {
            "name": "handle-request",
            "span_id": "span1",
            "service_name": "api",
            "duration_ms": 45.2,
            "error_flag": False,
        }
    ]
})

# Get analysis
anomalies = pulse.analyze_anomalies()
root_cause = pulse.analyze_root_cause()
graph = pulse.get_dependency_graph()
```

---

## Installation

### Prerequisites

- Python 3.10+
- Redis (optional, for distributed mode)
- Elasticsearch (optional, for analytics)

### Dependencies

```bash
# Core dependencies
pip install numpy pydantic

# OpenTelemetry
pip install opentelemetry-api \
             opentelemetry-sdk \
             opentelemetry-exporter-otlp

# Storage
pip install redis elasticsearch

# ML (optional but recommended)
pip install scikit-learn torch

# Web framework
pip install fastapi uvicorn
```

### Configuration

Create `config.yaml`:

```yaml
service:
  name: pulse
  host: 0.0.0.0
  port: 8080

storage:
  backend: sqlite
  path: ./data

ml:
  enabled: true
  models:
    isolation_forest:
      contamination: 0.1
    lstm:
      hidden_size: 64

alerting:
  enabled: true
  forecast_horizon_seconds: 300
```

---

## Usage

### Integrating with OpenTelemetry

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure OpenTelemetry
provider = TracerProvider()
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://localhost:4317")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Create spans
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("my-operation") as span:
    span.set_attribute("user.id", user_id)
    # Your code here
```

### Sending Traces to Pulse

```python
import httpx

# Send span directly
async with httpx.AsyncClient() as client:
    await client.post(
        "http://localhost:8080/api/v1/spans",
        json={
            "name": "my-span",
            "service_name": "my-service",
            "duration_ms": 123.45,
        }
    )
```

### Querying Traces

```python
# Get specific trace
trace = pulse.collector.get_trace("trace-id")

# List traces with filters
traces = pulse.collector.get_traces(
    service_name="api-gateway",
    limit=100,
)

# Search storage
results = pulse.storage.query_traces(
    QueryFilter(
        service_name="user-service",
        error_only=True,
        start_time=datetime.utcnow() - timedelta(hours=1),
    )
)
```

### Using ML Detection

```python
# Train models
pulse.train_models(historical_traces)

# Detect anomalies
score = pulse.anomaly_detector.detect(trace)

if score.is_anomaly:
    print(f"Anomaly detected! Score: {score.normalized:.2f}")
    print(f"Confidence: {score.confidence:.2f}")
    print(f"Features: {score.features}")
```

---

## ML Models

### Isolation Forest

**How it works:**
1. Build an ensemble of random decision trees
2. Anomalies get isolated in shorter paths
3. Calculate path length as anomaly score
4. Shorter paths = more anomalous

**Hyperparameters:**
```python
{
    "contamination": 0.1,      # Expected anomaly rate
    "n_estimators": 100,        # Number of trees
    "max_samples": 256,         # Samples per tree
    "random_state": 42,
}
```

**Best for:**
- High-dimensional data
- No labeled training data
- Fast inference needed

### LSTM Neural Network

**Architecture:**
- Input: Time-series features (20 dimensions)
- LSTM Layer 1: 64 units
- LSTM Layer 2: 64 units
- Dense Layer: 20 units
- Loss: MSE (reconstruction error)

**Training:**
```python
# Generate training data
traces, labels = generate_synthetic_traces(
    num_traces=1000,
    anomaly_rate=0.1,
)

# Train
detector.train(traces, epochs=10)

# Predict
score = detector.predict(trace)
```

**Best for:**
- Time-series patterns
- Sequence prediction
- Trend detection

### Ensemble Detection

**Combining predictions:**
```python
# Weighted average of scores
ensemble_score = (
    isolation_forest_score * 0.6 +
    lstm_score * 0.4
) * confidence_weights

# Anomaly if ensemble_score > threshold
is_anomaly = ensemble_score >= 0.7
```

---

## API Reference

### REST API Endpoints

#### Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/spans` | Ingest a single span |
| POST | `/api/v1/traces` | Ingest a complete trace |
| GET | `/api/v1/traces/{id}` | Get trace by ID |
| GET | `/api/v1/traces` | List traces |

#### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze/anomalies` | Detect anomalies |
| POST | `/api/v1/analyze/root-cause` | Root cause analysis |

#### Visualization

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/graphs/dependencies` | Get dependency graph |
| GET | `/api/v1/graphs/bottlenecks` | Get bottlenecks |

#### SLOs & Alerts

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/slo/status` | SLO compliance |
| GET | `/api/v1/alerts` | Active alerts |

### Request/Response Examples

#### Ingest Span
```bash
curl -X POST http://localhost:8080/api/v1/spans \
  -H "Content-Type: application/json" \
  -d '{
    "name": "db.query",
    "service_name": "user-service",
    "duration_ms": 25.3,
    "attributes": {
      "db.system": "postgresql",
      "db.statement": "SELECT * FROM users"
    }
  }'
```

#### Detect Anomalies
```bash
curl -X POST http://localhost:8080/api/v1/analyze/anomalies \
  -H "Content-Type: application/json" \
  -d '{"time_window_minutes": 60}'
```

Response:
```json
{
  "anomalies": [
    {
      "trace_id": "trace-123",
      "anomaly_score": 0.87,
      "confidence": 0.92,
      "features": {
        "error_rate": 0.15,
        "p95_duration_ms": 2500
      }
    }
  ],
  "total": 1
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | pulse | Service name |
| `STORAGE_BACKEND` | memory | Storage backend |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | localhost:4317 | OTLP endpoint |
| `LOG_LEVEL` | INFO | Logging level |

### Configuration File

See `config/collector.yaml` for full configuration options.

---

## Benchmarks

Performance benchmarks on standard hardware (8 cores, 16GB RAM):

| Component | Metric | Value |
|-----------|--------|-------|
| Collector | Throughput | 100,000 spans/sec |
| Storage (SQLite) | Write throughput | 10,000 traces/sec |
| Storage (SQLite) | Read latency | <1ms |
| ML Detection | Prediction latency | <10ms |
| Root Cause | Analysis time | <100ms |
| Graph Generation | Processing time | <500ms |

### Running Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/moggan1337/Pulse.git
cd Pulse

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run benchmarks
python benchmarks/run_benchmarks.py
```

### Project Structure

```
Pulse/
├── src/
│   ├── collector/     # Trace collection
│   ├── storage/       # Storage backends
│   ├── ml/            # ML models
│   ├── analysis/      # Root cause analysis
│   ├── alerting/      # Alerting system
│   ├── graphs/        # Dependency graphs
│   ├── slo/           # SLO tracking
│   ├── context/       # Context propagation
│   └── server.py      # HTTP API
├── tests/             # Test suite
├── benchmarks/        # Performance benchmarks
├── config/            # Configuration
├── docker/            # Docker files
└── docs/             # Documentation
```

---

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## License

Copyright 2024 Pulse Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

<p align="center">
  <strong>Built with ❤️ by the Pulse Team</strong>
</p>
