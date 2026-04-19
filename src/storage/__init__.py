"""Trace Storage Module with multiple backend support"""

import asyncio
import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator, TypeVar, Generic

try:
    import redis
    import msgpack
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from elasticsearch import Elasticsearch
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class StorageConfig:
    """Configuration for storage backends"""
    backend: str = "memory"  # memory, sqlite, redis, elasticsearch
    path: str = "./data"
    max_connections: int = 10
    ttl_seconds: int = 86400 * 7  # 7 days default retention
    batch_size: int = 100
    flush_interval: int = 5


@dataclass
class TraceIndex:
    """Index information for fast trace lookups"""
    trace_id: str
    start_time: datetime
    end_time: Optional[datetime]
    service_names: List[str]
    error_count: int
    total_spans: int
    duration_ms: float


@dataclass 
class QueryFilter:
    """Filter criteria for trace queries"""
    service_name: Optional[str] = None
    error_only: bool = False
    trace_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    attribute_filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 100
    offset: int = 0


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage backend"""
        pass
    
    @abstractmethod
    def store_trace(self, trace_data: Dict[str, Any]) -> bool:
        """Store a trace"""
        pass
    
    @abstractmethod
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a trace by ID"""
        pass
    
    @abstractmethod
    def query_traces(self, filter: QueryFilter) -> List[Dict[str, Any]]:
        """Query traces with filters"""
        pass
    
    @abstractmethod
    def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the storage backend"""
        pass


class MemoryStorage(StorageBackend):
    """
    In-memory storage backend for traces.
    
    Suitable for testing and small-scale deployments.
    Data is lost on restart.
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self._traces: Dict[str, Dict[str, Any]] = {}
        self._index: Dict[str, TraceIndex] = {}
        self._service_index: Dict[str, List[str]] = {}  # service -> trace_ids
        self._lock = asyncio.Lock()
        self._total_spans = 0
        self._total_errors = 0
    
    def initialize(self) -> None:
        """Initialize memory storage"""
        logger.info("Initializing memory storage backend")
    
    async def store_trace(self, trace_data: Dict[str, Any]) -> bool:
        """Store a trace in memory"""
        async with self._lock:
            trace_id = trace_data.get("trace_id")
            if not trace_id:
                return False
            
            # Index the trace
            spans = trace_data.get("spans", [])
            services = list(set(s.get("service_name", "unknown") for s in spans))
            errors = sum(1 for s in spans if s.get("error_flag", False))
            
            start_time_str = trace_data.get("start_time")
            end_time_str = trace_data.get("end_time")
            
            index = TraceIndex(
                trace_id=trace_id,
                start_time=datetime.fromisoformat(start_time_str) if start_time_str else datetime.utcnow(),
                end_time=datetime.fromisoformat(end_time_str) if end_time_str else None,
                service_names=services,
                error_count=errors,
                total_spans=len(spans),
                duration_ms=trace_data.get("total_duration_ms", 0.0),
            )
            
            self._traces[trace_id] = trace_data
            self._index[trace_id] = index
            
            # Update service index
            for service in services:
                if service not in self._service_index:
                    self._service_index[service] = []
                if trace_id not in self._service_index[service]:
                    self._service_index[service].append(trace_id)
            
            self._total_spans += len(spans)
            self._total_errors += errors
            
            return True
    
    def store_trace_sync(self, trace_data: Dict[str, Any]) -> bool:
        """Synchronous version of store_trace"""
        trace_id = trace_data.get("trace_id")
        if not trace_id:
            return False
        
        spans = trace_data.get("spans", [])
        services = list(set(s.get("service_name", "unknown") for s in spans))
        errors = sum(1 for s in spans if s.get("error_flag", False))
        
        start_time_str = trace_data.get("start_time")
        end_time_str = trace_data.get("end_time")
        
        index = TraceIndex(
            trace_id=trace_id,
            start_time=datetime.fromisoformat(start_time_str) if start_time_str else datetime.utcnow(),
            end_time=datetime.fromisoformat(end_time_str) if end_time_str else None,
            service_names=services,
            error_count=errors,
            total_spans=len(spans),
            duration_ms=trace_data.get("total_duration_ms", 0.0),
        )
        
        self._traces[trace_id] = trace_data
        self._index[trace_id] = index
        
        for service in services:
            if service not in self._service_index:
                self._service_index[service] = []
            if trace_id not in self._service_index[service]:
                self._service_index[service].append(trace_id)
        
        self._total_spans += len(spans)
        self._total_errors += errors
        
        return True
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace from memory"""
        return self._traces.get(trace_id)
    
    def query_traces(self, filter: QueryFilter) -> List[Dict[str, Any]]:
        """Query traces from memory"""
        results = []
        
        # Filter by service
        if filter.service_name:
            trace_ids = self._service_index.get(filter.service_name, [])
        else:
            trace_ids = list(self._index.keys())
        
        for trace_id in trace_ids:
            if trace_id not in self._index:
                continue
            
            index = self._index[trace_id]
            
            # Filter by error only
            if filter.error_only and index.error_count == 0:
                continue
            
            # Filter by time range
            if filter.start_time and index.start_time < filter.start_time:
                continue
            if filter.end_time and index.end_time and index.end_time > filter.end_time:
                continue
            
            # Filter by duration
            if filter.min_duration_ms and index.duration_ms < filter.min_duration_ms:
                continue
            if filter.max_duration_ms and index.duration_ms > filter.max_duration_ms:
                continue
            
            # Filter by trace ID
            if filter.trace_id and trace_id != filter.trace_id:
                continue
            
            # Get full trace data
            trace_data = self._traces.get(trace_id)
            if trace_data:
                results.append(trace_data)
        
        # Sort by start time descending
        results.sort(
            key=lambda t: t.get("start_time", ""),
            reverse=True
        )
        
        # Apply pagination
        return results[filter.offset:filter.offset + filter.limit]
    
    def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace from memory"""
        if trace_id not in self._traces:
            return False
        
        del self._traces[trace_id]
        
        if trace_id in self._index:
            index = self._index[trace_id]
            for service in index.service_names:
                if service in self._service_index:
                    try:
                        self._service_index[service].remove(trace_id)
                    except ValueError:
                        pass
            del self._index[trace_id]
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            "backend": "memory",
            "total_traces": len(self._traces),
            "total_spans": self._total_spans,
            "total_errors": self._total_errors,
            "indexed_services": len(self._service_index),
            "memory_usage_estimate_mb": len(json.dumps(self._traces)) / (1024 * 1024),
        }
    
    def close(self) -> None:
        """Close memory storage"""
        self._traces.clear()
        self._index.clear()
        self._service_index.clear()


class SQLiteStorage(StorageBackend):
    """
    SQLite-based storage backend for traces.
    
    Provides persistent storage with SQL querying capabilities.
    Suitable for single-node deployments.
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.db_path = Path(self.config.path) / "pulse_traces.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._buffer: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    def initialize(self) -> None:
        """Initialize SQLite database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        
        # Create tables
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trace_index (
                trace_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                services TEXT NOT NULL,
                error_count INTEGER DEFAULT 0,
                total_spans INTEGER DEFAULT 0,
                duration_ms REAL DEFAULT 0,
                FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
            )
        """)
        
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_start_time ON trace_index(start_time)
        """)
        
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_services ON trace_index(services)
        """)
        
        self._conn.commit()
        logger.info(f"SQLite storage initialized at {self.db_path}")
    
    async def store_trace(self, trace_data: Dict[str, Any]) -> bool:
        """Store a trace in SQLite"""
        async with self._lock:
            return self._store_trace_internal(trace_data)
    
    def _store_trace_internal(self, trace_data: Dict[str, Any]) -> bool:
        """Internal sync method to store trace"""
        trace_id = trace_data.get("trace_id")
        if not trace_id:
            return False
        
        spans = trace_data.get("spans", [])
        services = list(set(s.get("service_name", "unknown") for s in spans))
        errors = sum(1 for s in spans if s.get("error_flag", False))
        
        # Store full trace data
        data_json = json.dumps(trace_data)
        start_time = trace_data.get("start_time")
        end_time = trace_data.get("end_time")
        
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO traces (trace_id, data, start_time, end_time)
                VALUES (?, ?, ?, ?)
            """, (trace_id, data_json, start_time, end_time))
            
            self._conn.execute("""
                INSERT OR REPLACE INTO trace_index 
                (trace_id, start_time, end_time, services, error_count, total_spans, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                trace_id, 
                start_time, 
                end_time, 
                json.dumps(services),
                errors,
                len(spans),
                trace_data.get("total_duration_ms", 0.0)
            ))
            
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            return False
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace from SQLite"""
        cursor = self._conn.execute(
            "SELECT data FROM traces WHERE trace_id = ?",
            (trace_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return json.loads(row["data"])
        return None
    
    def query_traces(self, filter: QueryFilter) -> List[Dict[str, Any]]:
        """Query traces from SQLite"""
        conditions = []
        params = []
        
        if filter.service_name:
            conditions.append("services LIKE ?")
            params.append(f"%{filter.service_name}%")
        
        if filter.error_only:
            conditions.append("error_count > 0")
        
        if filter.start_time:
            conditions.append("start_time >= ?")
            params.append(filter.start_time.isoformat())
        
        if filter.end_time:
            conditions.append("(end_time IS NULL OR end_time <= ?)")
            params.append(filter.end_time.isoformat())
        
        if filter.min_duration_ms is not None:
            conditions.append("duration_ms >= ?")
            params.append(filter.min_duration_ms)
        
        if filter.max_duration_ms is not None:
            conditions.append("duration_ms <= ?")
            params.append(filter.max_duration_ms)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT data FROM traces t
            JOIN trace_index i ON t.trace_id = i.trace_id
            WHERE {where_clause}
            ORDER BY t.start_time DESC
            LIMIT ? OFFSET ?
        """
        params.extend([filter.limit, filter.offset])
        
        cursor = self._conn.execute(query, params)
        results = []
        
        for row in cursor.fetchall():
            results.append(json.loads(row["data"]))
        
        return results
    
    def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace from SQLite"""
        try:
            self._conn.execute("DELETE FROM trace_index WHERE trace_id = ?", (trace_id,))
            self._conn.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
            self._conn.commit()
            return True
        except sqlite3.Error:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        cursor = self._conn.execute("SELECT COUNT(*) as count FROM traces")
        total_traces = cursor.fetchone()["count"]
        
        cursor = self._conn.execute("SELECT SUM(total_spans) as total FROM trace_index")
        total_spans = cursor.fetchone()["total"] or 0
        
        cursor = self._conn.execute("SELECT SUM(error_count) as total FROM trace_index")
        total_errors = cursor.fetchone()["total"] or 0
        
        return {
            "backend": "sqlite",
            "db_path": str(self.db_path),
            "total_traces": total_traces,
            "total_spans": total_spans,
            "total_errors": total_errors,
        }
    
    def close(self) -> None:
        """Close SQLite connection"""
        if self._conn:
            self._conn.close()
            self._conn = None


class RedisStorage(StorageBackend):
    """
    Redis-based storage backend for traces.
    
    Provides high-performance distributed storage with TTL support.
    Requires Redis server.
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required for RedisStorage")
        
        self.config = config or StorageConfig()
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._ttl = config.ttl_seconds if config else 86400 * 7
    
    def initialize(self) -> None:
        """Initialize Redis connection"""
        self._client = redis.Redis(
            host=self.config.path or "localhost",
            port=6379,
            db=0,
            decode_responses=True,
        )
        logger.info("Redis storage initialized")
    
    async def store_trace(self, trace_data: Dict[str, Any]) -> bool:
        """Store a trace in Redis"""
        trace_id = trace_data.get("trace_id")
        if not trace_id:
            return False
        
        # Store trace data as JSON
        key = f"trace:{trace_id}"
        self._client.setex(key, self._ttl, json.dumps(trace_data))
        
        # Index by service
        spans = trace_data.get("spans", [])
        services = set(s.get("service_name", "unknown") for s in spans)
        
        for service in services:
            self._client.sadd(f"service:{service}:traces", trace_id)
            self._client.expire(f"service:{service}:traces", self._ttl)
        
        # Index by time
        start_time = trace_data.get("start_time")
        if start_time:
            self._client.zadd("traces:by_time", {trace_id: float(datetime.fromisoformat(start_time).timestamp())})
        
        return True
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace from Redis"""
        key = f"trace:{trace_id}"
        data = self._client.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    def query_traces(self, filter: QueryFilter) -> List[Dict[str, Any]]:
        """Query traces from Redis"""
        if filter.service_name:
            trace_ids = self._client.smembers(f"service:{filter.service_name}:traces")
        else:
            trace_ids = self._client.zrange("traces:by_time", 0, -1)
        
        results = []
        for trace_id in trace_ids:
            trace = self.get_trace(trace_id)
            if trace:
                # Apply additional filters
                if filter.error_only:
                    if not any(s.get("error_flag") for s in trace.get("spans", [])):
                        continue
                
                if filter.start_time:
                    trace_start = datetime.fromisoformat(trace.get("start_time", "1970-01-01"))
                    if trace_start < filter.start_time:
                        continue
                
                if filter.end_time:
                    trace_end = datetime.fromisoformat(trace.get("end_time", "2038-01-01"))
                    if trace_end > filter.end_time:
                        continue
                
                results.append(trace)
        
        # Sort and paginate
        results.sort(key=lambda t: t.get("start_time", ""), reverse=True)
        return results[filter.offset:filter.offset + filter.limit]
    
    def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace from Redis"""
        key = f"trace:{trace_id}"
        return bool(self._client.delete(key))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            "backend": "redis",
            "total_traces": self._client.zcard("traces:by_time"),
            "memory_usage": self._client.info("memory").get("used_memory_human", "unknown"),
        }
    
    def close(self) -> None:
        """Close Redis connection"""
        if self._client:
            self._client.close()


class TraceStorage:
    """
    High-level trace storage interface with backend abstraction.
    
    Provides a unified API for storing and querying traces across
    different storage backends.
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self._backend: Optional[StorageBackend] = None
        self._initialize_backend()
    
    def _initialize_backend(self) -> None:
        """Initialize the appropriate storage backend"""
        backend_map = {
            "memory": MemoryStorage,
            "sqlite": SQLiteStorage,
            "redis": RedisStorage,
        }
        
        backend_class = backend_map.get(self.config.backend, MemoryStorage)
        
        try:
            self._backend = backend_class(self.config)
            self._backend.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize {self.config.backend}, falling back to memory: {e}")
            self._backend = MemoryStorage(self.config)
            self._backend.initialize()
    
    async def store_trace(self, trace_data: Dict[str, Any]) -> bool:
        """Store a trace"""
        return await self._backend.store_trace(trace_data)
    
    def store_trace_sync(self, trace_data: Dict[str, Any]) -> bool:
        """Synchronous store trace"""
        if hasattr(self._backend, "store_trace_sync"):
            return self._backend.store_trace_sync(trace_data)
        return asyncio.get_event_loop().run_until_complete(
            self._backend.store_trace(trace_data)
        )
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace by ID"""
        return self._backend.get_trace(trace_id)
    
    def query_traces(self, filter: Optional[QueryFilter] = None) -> List[Dict[str, Any]]:
        """Query traces with filters"""
        if filter is None:
            filter = QueryFilter()
        return self._backend.query_traces(filter)
    
    def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace"""
        return self._backend.delete_trace(trace_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return self._backend.get_stats()
    
    def close(self) -> None:
        """Close storage backend"""
        self._backend.close()


# Aggregation utilities
class TraceAggregator:
    """
    Aggregates trace data for analytics and reporting.
    """
    
    def __init__(self, storage: TraceStorage):
        self.storage = storage
    
    def aggregate_by_service(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics by service"""
        filter = QueryFilter(start_time=start_time, end_time=end_time, limit=10000)
        traces = self.storage.query_traces(filter)
        
        service_metrics: Dict[str, Dict[str, Any]] = {}
        
        for trace in traces:
            for span in trace.get("spans", []):
                service = span.get("service_name", "unknown")
                
                if service not in service_metrics:
                    service_metrics[service] = {
                        "total_spans": 0,
                        "total_errors": 0,
                        "total_duration_ms": 0.0,
                        "min_duration_ms": float("inf"),
                        "max_duration_ms": 0.0,
                    }
                
                metrics = service_metrics[service]
                metrics["total_spans"] += 1
                
                if span.get("error_flag"):
                    metrics["total_errors"] += 1
                
                duration = span.get("duration_ms", 0)
                metrics["total_duration_ms"] += duration
                metrics["min_duration_ms"] = min(metrics["min_duration_ms"], duration)
                metrics["max_duration_ms"] = max(metrics["max_duration_ms"], duration)
        
        # Calculate averages
        for service, metrics in service_metrics.items():
            count = metrics["total_spans"]
            if count > 0:
                metrics["avg_duration_ms"] = metrics["total_duration_ms"] / count
                metrics["error_rate"] = metrics["total_errors"] / count
            else:
                metrics["avg_duration_ms"] = 0
                metrics["error_rate"] = 0
        
        return service_metrics
    
    def aggregate_by_time_window(
        self,
        window_minutes: int = 5,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by time window"""
        filter = QueryFilter(start_time=start_time, end_time=end_time, limit=10000)
        traces = self.storage.query_traces(filter)
        
        windows: Dict[str, Dict[str, Any]] = {}
        
        for trace in traces:
            trace_start = trace.get("start_time")
            if not trace_start:
                continue
            
            dt = datetime.fromisoformat(trace_start)
            window_key = dt.strftime("%Y-%m-%d %H:%M")
            
            if window_key not in windows:
                windows[window_key] = {
                    "window_start": window_key,
                    "total_traces": 0,
                    "total_spans": 0,
                    "total_errors": 0,
                    "total_duration_ms": 0.0,
                }
            
            w = windows[window_key]
            w["total_traces"] += 1
            w["total_spans"] += len(trace.get("spans", []))
            w["total_errors"] += sum(1 for s in trace.get("spans", []) if s.get("error_flag"))
            w["total_duration_ms"] += trace.get("total_duration_ms", 0)
        
        # Calculate averages and sort
        result = []
        for window_data in windows.values():
            if window_data["total_traces"] > 0:
                window_data["avg_trace_duration_ms"] = (
                    window_data["total_duration_ms"] / window_data["total_traces"]
                )
                window_data["error_rate"] = (
                    window_data["total_errors"] / window_data["total_spans"]
                )
            result.append(window_data)
        
        result.sort(key=lambda w: w["window_start"])
        return result
    
    def get_percentiles(
        self,
        service_name: Optional[str] = None,
        metric: str = "duration_ms",
        percentiles: List[int] = None,
    ) -> Dict[str, float]:
        """Calculate percentiles for a metric"""
        if percentiles is None:
            percentiles = [50, 90, 95, 99]
        
        filter = QueryFilter(service_name=service_name, limit=10000)
        traces = self.storage.query_traces(filter)
        
        values = []
        for trace in traces:
            for span in trace.get("spans", []):
                if service_name and span.get("service_name") != service_name:
                    continue
                value = span.get(metric, 0)
                if value:
                    values.append(value)
        
        values.sort()
        n = len(values)
        
        result = {}
        for p in percentiles:
            idx = int(n * p / 100)
            if idx >= n:
                idx = n - 1
            result[f"p{p}"] = values[idx] if n > 0 else 0
        
        return result
