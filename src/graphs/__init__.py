"""Dependency Graph Generation Module"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the dependency graph"""
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class EdgeType(Enum):
    """Types of edges in the dependency graph"""
    HTTP = "http"
    GRPC = "grpc"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    ASYNC = "async"
    UNKNOWN = "unknown"


@dataclass
class GraphNode:
    """Represents a node in the dependency graph"""
    id: str
    name: str
    node_type: NodeType
    
    # Metadata
    version: Optional[str] = None
    environment: Optional[str] = None
    
    # Metrics (aggregated)
    request_count: int = 0
    error_count: int = 0
    avg_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    
    # Health
    health_score: float = 1.0
    is_healthy: bool = True
    
    # Position for visualization
    x: float = 0.0
    y: float = 0.0
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(1, self.request_count)
    
    @property
    def success_rate(self) -> float:
        return 1.0 - self.error_rate
    
    def update_health(self) -> None:
        """Update health score based on metrics"""
        if self.request_count == 0:
            self.health_score = 1.0
            self.is_healthy = True
            return
        
        # Calculate health based on error rate and latency
        error_health = 1.0 - min(self.error_rate * 10, 1.0)  # 10% error = 0 health
        latency_health = 1.0 if self.p95_duration_ms < 1000 else max(0, 1 - (self.p95_duration_ms - 1000) / 10000)
        
        self.health_score = (error_health + latency_health) / 2
        self.is_healthy = self.health_score > 0.5


@dataclass
class GraphEdge:
    """Represents an edge (dependency) in the graph"""
    source: str
    target: str
    edge_type: EdgeType
    
    # Metrics
    call_count: int = 0
    error_count: int = 0
    avg_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    
    # Health
    health_score: float = 1.0
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(1, self.call_count)
    
    def update_health(self) -> None:
        """Update health score based on metrics"""
        if self.call_count == 0:
            self.health_score = 1.0
            return
        
        error_health = 1.0 - min(self.error_rate * 10, 1.0)
        latency_health = 1.0 if self.p95_duration_ms < 500 else max(0, 1 - (self.p95_duration_ms - 500) / 2000)
        
        self.health_score = (error_health + latency_health) / 2


@dataclass
class DependencyGraph:
    """Complete dependency graph"""
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], GraphEdge] = field(default_factory=dict)
    
    # Graph metadata
    name: str = "service-graph"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Statistics
    total_calls: int = 0
    total_errors: int = 0
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.updated_at = datetime.utcnow()
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph"""
        key = (edge.source, edge.target)
        
        if key in self.edges:
            # Update existing edge
            existing = self.edges[key]
            # Weighted average
            total_calls = existing.call_count + edge.call_count
            if total_calls > 0:
                existing.avg_duration_ms = (
                    (existing.avg_duration_ms * existing.call_count +
                     edge.avg_duration_ms * edge.call_count) / total_calls
                )
                existing.error_count = existing.error_count + edge.error_count
            existing.call_count = total_calls
            existing.total_duration_ms += edge.total_duration_ms
        else:
            self.edges[key] = edge
        
        self.total_calls += edge.call_count
        self.total_errors += edge.error_count
        self.updated_at = datetime.utcnow()
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """Get all edges from a node"""
        return [e for (s, t), e in self.edges.items() if s == node_id]
    
    def get_edges_to(self, node_id: str) -> List[GraphEdge]:
        """Get all edges to a node"""
        return [e for (s, t), e in self.edges.items() if t == node_id]
    
    def get_dependencies(self, node_id: str) -> List[str]:
        """Get direct dependencies of a node"""
        return [t for (s, t), e in self.edges.items() if s == node_id]
    
    def get_dependents(self, node_id: str) -> List[str]:
        """Get direct dependents of a node"""
        return [s for (s, t), e in self.edges.items() if t == node_id]
    
    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find a path between two nodes using BFS"""
        if source == target:
            return [source]
        
        visited = {source}
        queue = [(source, [source])]
        
        while queue:
            current, path = queue.pop(0)
            
            for neighbor in self.get_dependencies(current):
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_critical_path(self) -> List[str]:
        """Get the critical path based on latency"""
        if not self.edges:
            return []
        
        # Find path with highest total latency
        best_path = []
        max_latency = 0.0
        
        for start_node in self.nodes:
            for end_node in self.nodes:
                if start_node != end_node:
                    path = self.find_path(start_node, end_node)
                    if path:
                        total_latency = sum(
                            self.edges.get((path[i], path[i+1]), GraphEdge(path[i], path[i+1], EdgeType.UNKNOWN)).avg_duration_ms
                            for i in range(len(path) - 1)
                        )
                        if total_latency > max_latency:
                            max_latency = total_latency
                            best_path = path
        
        return best_path
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of the graph"""
        node_health = [n.health_score for n in self.nodes.values()]
        edge_health = [e.health_score for e in self.edges.values()]
        
        unhealthy_nodes = [n.id for n in self.nodes.values() if not n.is_healthy]
        unhealthy_edges = [
            f"{s}->{t}" for (s, t), e in self.edges.items()
            if e.health_score < 0.5
        ]
        
        return {
            "overall_health": sum(node_health) / max(1, len(node_health)),
            "avg_node_health": sum(node_health) / max(1, len(node_health)),
            "avg_edge_health": sum(edge_health) / max(1, len(edge_health)),
            "unhealthy_nodes": unhealthy_nodes,
            "unhealthy_edges": unhealthy_edges,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary"""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "nodes": {
                node_id: {
                    "id": node.id,
                    "name": node.name,
                    "type": node.node_type.value,
                    "version": node.version,
                    "metrics": {
                        "request_count": node.request_count,
                        "error_count": node.error_count,
                        "avg_duration_ms": node.avg_duration_ms,
                        "p95_duration_ms": node.p95_duration_ms,
                    },
                    "health_score": node.health_score,
                    "is_healthy": node.is_healthy,
                    "metadata": node.metadata,
                }
                for node_id, node in self.nodes.items()
            },
            "edges": {
                f"{source}->{target}": {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.edge_type.value,
                    "call_count": edge.call_count,
                    "error_count": edge.error_count,
                    "avg_duration_ms": edge.avg_duration_ms,
                    "health_score": edge.health_score,
                }
                for (source, target), edge in self.edges.items()
            },
            "stats": {
                "total_calls": self.total_calls,
                "total_errors": self.total_errors,
            },
        }
    
    def to_cytoscape_json(self) -> Dict[str, Any]:
        """Convert to Cytoscape.js format for visualization"""
        nodes = []
        for node_id, node in self.nodes.items():
            color = "#22c55e"  # green
            if node.health_score < 0.8:
                color = "#eab308"  # yellow
            if node.health_score < 0.5:
                color = "#ef4444"  # red
            
            nodes.append({
                "data": {
                    "id": node_id,
                    "label": node.name,
                    "type": node.node_type.value,
                    "health": node.health_score,
                    "color": color,
                    "metrics": {
                        "requests": node.request_count,
                        "errors": node.error_count,
                        "p95": node.p95_duration_ms,
                    },
                }
            })
        
        edges = []
        for (source, target), edge in self.edges.items():
            color = "#94a3b8"  # gray
            if edge.health_score < 0.8:
                color = "#eab308"
            if edge.health_score < 0.5:
                color = "#ef4444"
            
            edges.append({
                "data": {
                    "id": f"{source}-{target}",
                    "source": source,
                    "target": target,
                    "type": edge.edge_type.value,
                    "weight": edge.call_count,
                    "color": color,
                }
            })
        
        return {"nodes": nodes, "edges": edges}


class DependencyGraphGenerator:
    """
    Generates dependency graphs from trace data.
    
    Analyzes traces to build a service dependency graph showing:
    - Service relationships
    - Communication patterns
    - Latency and error metrics
    - Health status
    """
    
    def __init__(self):
        self.graph = DependencyGraph()
        self.trace_history: List[Dict[str, Any]] = []
    
    def generate_from_traces(
        self,
        traces: List[Dict[str, Any]],
        time_window: Optional[timedelta] = None,
    ) -> DependencyGraph:
        """
        Generate dependency graph from traces.
        
        Args:
            traces: List of trace dictionaries
            time_window: Optional time window to filter traces
        
        Returns:
            DependencyGraph with nodes and edges
        """
        self.graph = DependencyGraph()
        self.trace_history = traces
        
        # Filter by time window if specified
        if time_window:
            cutoff = datetime.utcnow() - time_window
            traces = self._filter_traces_by_time(traces, cutoff)
        
        # Extract nodes and edges from traces
        for trace in traces:
            self._process_trace(trace)
        
        # Update all health scores
        for node in self.graph.nodes.values():
            node.update_health()
        
        for edge in self.graph.edges.values():
            edge.update_health()
        
        return self.graph
    
    def _filter_traces_by_time(
        self,
        traces: List[Dict[str, Any]],
        cutoff: datetime,
    ) -> List[Dict[str, Any]]:
        """Filter traces by time window"""
        filtered = []
        
        for trace in traces:
            start_time_str = trace.get("start_time")
            if start_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str)
                    if start_time >= cutoff:
                        filtered.append(trace)
                except ValueError:
                    filtered.append(trace)
            else:
                filtered.append(trace)
        
        return filtered
    
    def _process_trace(self, trace: Dict[str, Any]) -> None:
        """Process a single trace to extract graph elements"""
        spans = trace.get("spans", [])
        if not spans:
            return
        
        # Build span map for parent lookups
        span_map = {s.get("span_id"): s for s in spans}
        
        # Track which services we've seen
        seen_services: Set[str] = set()
        
        for span in spans:
            # Extract service info
            service_name = span.get("service_name", "unknown")
            service_id = service_name.lower().replace(" ", "-")
            
            seen_services.add(service_id)
            
            # Add or update node
            if service_id not in self.graph.nodes:
                self.graph.add_node(GraphNode(
                    id=service_id,
                    name=service_name,
                    node_type=self._infer_node_type(service_name),
                ))
            
            node = self.graph.nodes[service_id]
            node.request_count += 1
            node.avg_duration_ms = (
                (node.avg_duration_ms * (node.request_count - 1) +
                 span.get("duration_ms", 0)) / node.request_count
            )
            
            if span.get("error_flag"):
                node.error_count += 1
            
            # Track p95 duration
            duration = span.get("duration_ms", 0)
            if duration > node.p95_duration_ms:
                node.p95_duration_ms = duration
            
            # Extract dependencies from parent-child relationships
            parent_span_id = span.get("parent_span_id")
            if parent_span_id and parent_span_id in span_map:
                parent_span = span_map[parent_span_id]
                parent_service = parent_span.get("service_name", "unknown")
                parent_id = parent_service.lower().replace(" ", "-")
                
                # Add parent if not seen
                if parent_id not in self.graph.nodes:
                    self.graph.add_node(GraphNode(
                        id=parent_id,
                        name=parent_service,
                        node_type=self._infer_node_type(parent_service),
                    ))
                
                # Add or update edge
                edge_key = (parent_id, service_id)
                
                if edge_key not in self.graph.edges:
                    self.graph.add_edge(GraphEdge(
                        source=parent_id,
                        target=service_id,
                        edge_type=self._infer_edge_type(span),
                    ))
                
                edge = self.graph.edges[edge_key]
                edge.call_count += 1
                edge.total_duration_ms += duration
                edge.avg_duration_ms = edge.total_duration_ms / edge.call_count
                
                if span.get("error_flag"):
                    edge.error_count += 1
                
                if duration > edge.p95_duration_ms:
                    edge.p95_duration_ms = duration
    
    def _infer_node_type(self, name: str) -> NodeType:
        """Infer node type from service name"""
        name_lower = name.lower()
        
        if any(db in name_lower for db in ["db", "database", "postgres", "mysql", "mongodb"]):
            return NodeType.DATABASE
        if any(c in name_lower for c in ["cache", "redis", "memcached"]):
            return NodeType.CACHE
        if any(q in name_lower for q in ["queue", "kafka", "rabbit", "nats"]):
            return NodeType.QUEUE
        if any(e in name_lower for e in ["external", "api", "third-party"]):
            return NodeType.EXTERNAL
        
        return NodeType.SERVICE
    
    def _infer_edge_type(self, span: Dict[str, Any]) -> EdgeType:
        """Infer edge type from span attributes"""
        attributes = span.get("attributes", {})
        
        # Check for RPC type
        rpc_type = attributes.get("rpc.system")
        if rpc_type == "grpc":
            return EdgeType.GRPC
        
        # Check for HTTP
        http_method = attributes.get("http.method")
        if http_method:
            return EdgeType.HTTP
        
        # Check for database
        db_system = attributes.get("db.system")
        if db_system:
            return EdgeType.DATABASE
        
        # Check for messaging
        messaging_system = attributes.get("messaging.system")
        if messaging_system:
            return EdgeType.MESSAGE_QUEUE
        
        return EdgeType.UNKNOWN
    
    def update_with_realtime_span(
        self,
        span: Dict[str, Any],
        parent_span: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update graph with a single real-time span.
        
        Used for live graph updates without reprocessing all traces.
        """
        service_name = span.get("service_name", "unknown")
        service_id = service_name.lower().replace(" ", "-")
        
        # Update or create node
        if service_id not in self.graph.nodes:
            self.graph.add_node(GraphNode(
                id=service_id,
                name=service_name,
                node_type=self._infer_node_type(service_name),
            ))
        
        node = self.graph.nodes[service_id]
        node.request_count += 1
        duration = span.get("duration_ms", 0)
        node.avg_duration_ms = (
            (node.avg_duration_ms * (node.request_count - 1) + duration) / node.request_count
        )
        
        if span.get("error_flag"):
            node.error_count += 1
        
        if duration > node.p95_duration_ms:
            node.p95_duration_ms = duration
        
        node.update_health()
        
        # Update edge if parent exists
        if parent_span:
            parent_service = parent_span.get("service_name", "unknown")
            parent_id = parent_service.lower().replace(" ", "-")
            
            edge_key = (parent_id, service_id)
            
            if edge_key not in self.graph.edges:
                self.graph.add_edge(GraphEdge(
                    source=parent_id,
                    target=service_id,
                    edge_type=self._infer_edge_type(span),
                ))
            
            edge = self.graph.edges[edge_key]
            edge.call_count += 1
            edge.total_duration_ms += duration
            edge.avg_duration_ms = edge.total_duration_ms / edge.call_count
            
            if span.get("error_flag"):
                edge.error_count += 1
            
            if duration > edge.p95_duration_ms:
                edge.p95_duration_ms = duration
            
            edge.update_health()
        
        self.graph.updated_at = datetime.utcnow()
    
    def get_service_dependencies(
        self,
        service_name: str,
        depth: int = 1,
    ) -> Dict[str, Any]:
        """
        Get dependencies for a service.
        
        Args:
            service_name: Name of the service
            depth: How many levels of dependencies to include
        
        Returns:
            Dictionary with dependency tree
        """
        service_id = service_name.lower().replace(" ", "-")
        
        if service_id not in self.graph.nodes:
            return {"error": "Service not found"}
        
        def build_tree(node_id: str, current_depth: int) -> Dict[str, Any]:
            if current_depth > depth:
                return {}
            
            node = self.graph.nodes.get(node_id)
            if not node:
                return {}
            
            result = {
                "name": node.name,
                "type": node.node_type.value,
                "health": node.health_score,
                "metrics": {
                    "requests": node.request_count,
                    "errors": node.error_count,
                    "p95": node.p95_duration_ms,
                },
            }
            
            if current_depth < depth:
                result["dependencies"] = {}
                for dep_id in self.graph.get_dependencies(node_id):
                    result["dependencies"][dep_id] = build_tree(dep_id, current_depth + 1)
            
            return result
        
        return build_tree(service_id, 0)
    
    def find_bottlenecks(self) -> List[Dict[str, Any]]:
        """Find potential bottlenecks in the dependency graph"""
        bottlenecks = []
        
        # Find high-latency edges
        for (source, target), edge in self.graph.edges.items():
            if edge.p95_duration_ms > 500:  # 500ms threshold
                bottlenecks.append({
                    "type": "high_latency",
                    "source": source,
                    "target": target,
                    "p95_duration_ms": edge.p95_duration_ms,
                    "health_score": edge.health_score,
                })
        
        # Find nodes with high fan-out
        for node_id, node in self.graph.nodes.items():
            dependents = self.graph.get_dependents(node_id)
            dependencies = self.graph.get_dependencies(node_id)
            
            if len(dependents) > 10:  # High fan-out
                bottlenecks.append({
                    "type": "high_fan_out",
                    "node": node_id,
                    "dependents_count": len(dependents),
                    "impact": "failure would affect many services",
                })
            
            if len(dependencies) > 10:  # High fan-in
                bottlenecks.append({
                    "type": "high_fan_in",
                    "node": node_id,
                    "dependencies_count": len(dependencies),
                    "impact": "many dependencies could affect this service",
                })
        
        # Find unhealthy nodes
        for node_id, node in self.graph.nodes.items():
            if not node.is_healthy:
                bottlenecks.append({
                    "type": "unhealthy",
                    "node": node_id,
                    "health_score": node.health_score,
                    "error_rate": node.error_rate,
                })
        
        return bottlenecks
    
    def export_for_visualization(self) -> Dict[str, Any]:
        """Export graph data for visualization tools"""
        return self.graph.to_cytoscape_json()


def create_graph_generator() -> DependencyGraphGenerator:
    """Create a new dependency graph generator"""
    return DependencyGraphGenerator()
