"""Service Level Objectives (SLO) Tracking Module"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SLOStatus(Enum):
    """SLO compliance status"""
    HEALTHY = "healthy"
    AT_RISK = "at_risk"
    BREACHING = "breaching"
    UNKNOWN = "unknown"


class SLIType(Enum):
    """Service Level Indicator types"""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    CORRECTNESS = "correctness"
    THROUGHPUT = "throughput"
    QUALITY = "quality"


@dataclass
class SLOTarget:
    """Defines an SLO target"""
    slo_id: str
    name: str
    description: str
    
    # Service scope
    service_name: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    
    # SLI configuration
    sli_type: SLIType
    target_value: float  # e.g., 0.999 for 99.9% availability
    window: timedelta = timedelta(days=30)
    
    # Thresholds
    at_risk_threshold: float = 0.95  # 95% of target
    breaching_threshold: float = 0.90  # 90% of target
    
    # Budget
    error_budget_percent: float = 1.0  # 1% error budget
    burn_rate_threshold: float = 1.0  # How fast we can burn budget
    
    def target_to_percent(self) -> float:
        """Convert target value to percentage"""
        if self.sli_type == SLIType.AVAILABILITY:
            return self.target_value * 100
        elif self.sli_type == SLIType.LATENCY:
            return self.target_value  # Already in ms
        return self.target_value * 100
    
    def get_at_risk_value(self) -> float:
        """Get the at-risk threshold value"""
        return self.target_value * self.at_risk_threshold
    
    def get_breaching_value(self) -> float:
        """Get the breaching threshold value"""
        return self.target_value * self.breaching_threshold


@dataclass
class SLOCompliance:
    """Current SLO compliance status"""
    slo_id: str
    current_value: float
    target_value: float
    
    # Status
    status: SLOStatus
    
    # Time windows
    window_start: datetime
    window_end: datetime
    
    # Metrics
    total_requests: int = 0
    good_requests: int = 0
    bad_requests: int = 0
    
    # Error budget
    error_budget_remaining: float = 100.0
    error_budget_used: float = 0.0
    burn_rate: float = 0.0
    
    # Predictions
    time_to_budget_exhaustion: Optional[float] = None  # in hours
    projected_compliance: float = 0.0
    
    # History
    historical_compliance: List[float] = field(default_factory=list)
    
    @property
    def compliance_percent(self) -> float:
        """Get compliance as percentage"""
        return self.current_value * 100
    
    @property
    def is_compliant(self) -> bool:
        """Check if currently meeting SLO"""
        return self.status == SLOStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "slo_id": self.slo_id,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "compliance_percent": self.compliance_percent,
            "status": self.status.value,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "total_requests": self.total_requests,
            "good_requests": self.good_requests,
            "bad_requests": self.bad_requests,
            "error_budget_remaining": self.error_budget_remaining,
            "error_budget_used": self.error_budget_used,
            "burn_rate": self.burn_rate,
            "time_to_budget_exhaustion": self.time_to_budget_exhaustion,
            "projected_compliance": self.projected_compliance,
        }


@dataclass
class SLOHistoryEntry:
    """Historical SLO data point"""
    timestamp: datetime
    compliance: float
    request_count: int
    error_count: int
    p95_latency: float


class SLITracker:
    """
    Tracks Service Level Indicators (SLIs) from trace data.
    """
    
    def __init__(self):
        # Time series storage for SLI metrics
        self.availability_history: deque = deque(maxlen=10000)
        self.latency_history: deque = deque(maxlen=10000)
        
        # Current window data
        self.current_window_start: Optional[datetime] = None
        self.window_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "good": 0,
            "bad": 0,
            "total": 0,
            "latencies": [],
        })
    
    def record_request(
        self,
        service_name: str,
        is_good: bool,
        latency_ms: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a request for SLI tracking"""
        ts = timestamp or datetime.utcnow()
        
        # Initialize window if needed
        if self.current_window_start is None:
            self.current_window_start = ts
        
        # Update window data
        self.window_data[service_name]["total"] += 1
        if is_good:
            self.window_data[service_name]["good"] += 1
        else:
            self.window_data[service_name]["bad"] += 1
        self.window_data[service_name]["latencies"].append(latency_ms)
        
        # Record history
        self.availability_history.append((ts, service_name, is_good))
        self.latency_history.append((ts, service_name, latency_ms))
    
    def calculate_availability(
        self,
        service_name: Optional[str] = None,
        window: Optional[timedelta] = None,
    ) -> float:
        """Calculate availability for a service"""
        cutoff = datetime.utcnow() - window if window else datetime.min
        
        good = 0
        total = 0
        
        for ts, svc, is_good in self.availability_history:
            if ts < cutoff:
                continue
            if service_name and svc != service_name:
                continue
            total += 1
            if is_good:
                good += 1
        
        return good / total if total > 0 else 1.0
    
    def calculate_latency_sli(
        self,
        service_name: Optional[str] = None,
        window: Optional[timedelta] = None,
        threshold_ms: float = 1000,
    ) -> float:
        """
        Calculate latency SLI (percentage of requests under threshold).
        """
        cutoff = datetime.utcnow() - window if window else datetime.min
        
        total = 0
        good = 0
        
        for ts, svc, latency in self.latency_history:
            if ts < cutoff:
                continue
            if service_name and svc != service_name:
                continue
            total += 1
            if latency <= threshold_ms:
                good += 1
        
        return good / total if total > 0 else 1.0
    
    def get_current_metrics(self, service_name: str) -> Dict[str, float]:
        """Get current window metrics for a service"""
        data = self.window_data.get(service_name, {"good": 0, "bad": 0, "total": 0, "latencies": []})
        
        total = data["total"]
        good = data["good"]
        latencies = data["latencies"]
        
        return {
            "availability": good / total if total > 0 else 1.0,
            "request_count": total,
            "error_count": data["bad"],
            "p50_latency": np.percentile(latencies, 50) if latencies else 0,
            "p95_latency": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency": np.percentile(latencies, 99) if latencies else 0,
        }
    
    def reset_window(self) -> None:
        """Reset the current window"""
        self.current_window_start = datetime.utcnow()
        self.window_data.clear()


class SLOTracker:
    """
    Main SLO tracking system.
    
    Tracks multiple SLOs, calculates compliance, monitors error budgets,
    and predicts when SLOs will be breached.
    """
    
    def __init__(self):
        self.slis = SLITracker()
        self.slos: Dict[str, SLOTarget] = {}
        self.compliance: Dict[str, SLOCompliance] = {}
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def register_slo(self, slo: SLOTarget) -> None:
        """Register a new SLO"""
        self.slos[slo.slo_id] = slo
        logger.info(f"Registered SLO: {slo.name} ({slo.slo_id})")
    
    def remove_slo(self, slo_id: str) -> bool:
        """Remove an SLO"""
        if slo_id in self.slos:
            del self.slos[slo_id]
            return True
        return False
    
    def record_trace(self, trace: Dict[str, Any]) -> None:
        """
        Record trace data for SLO tracking.
        
        Evaluates each span against registered SLOs.
        """
        spans = trace.get("spans", [])
        trace_start = trace.get("start_time")
        
        if trace_start:
            try:
                ts = datetime.fromisoformat(trace_start)
            except ValueError:
                ts = datetime.utcnow()
        else:
            ts = datetime.utcnow()
        
        for span in spans:
            service_name = span.get("service_name", "unknown")
            duration = span.get("duration_ms", 0)
            is_error = span.get("error_flag", False)
            
            # Record for all matching SLOs
            for slo in self.slos.values():
                if slo.service_name and slo.service_name != service_name:
                    continue
                
                # Determine if request is "good" for this SLO
                is_good = True
                
                if slo.sli_type == SLIType.AVAILABILITY:
                    is_good = not is_error
                elif slo.sli_type == SLIType.LATENCY:
                    is_good = duration <= slo.target_value
                
                self.slis.record_request(service_name, is_good, duration, ts)
    
    def evaluate_slo(self, slo_id: str) -> Optional[SLOCompliance]:
        """Evaluate current compliance for an SLO"""
        if slo_id not in self.slos:
            return None
        
        slo = self.slos[slo_id]
        
        # Calculate current SLI value
        if slo.sli_type == SLIType.AVAILABILITY:
            current_value = self.slis.calculate_availability(
                slo.service_name,
                slo.window,
            )
        elif slo.sli_type == SLIType.LATENCY:
            current_value = self.slis.calculate_latency_sli(
                slo.service_name,
                slo.window,
                slo.target_value,
            )
        else:
            current_value = self.slis.calculate_availability(
                slo.service_name,
                slo.window,
            )
        
        # Calculate error budget
        window_start = datetime.utcnow() - slo.window
        window_end = datetime.utcnow()
        
        # Get metrics for budget calculation
        metrics = self.slis.get_current_metrics(slo.service_name or "")
        total_requests = metrics.get("request_count", 1)
        good_requests = int(total_requests * current_value)
        bad_requests = total_requests - good_requests
        
        # Calculate error budget
        max_allowed_bad = total_requests * (1 - slo.target_value)
        error_budget_remaining = max(0, (max_allowed_bad - bad_requests) / max(1, max_allowed_bad) * 100) if max_allowed_bad > 0 else 100
        error_budget_used = 100 - error_budget_remaining
        
        # Calculate burn rate
        expected_bad_rate = 1 - slo.target_value
        actual_bad_rate = bad_requests / max(1, total_requests)
        burn_rate = actual_bad_rate / expected_bad_rate if expected_bad_rate > 0 else 0
        
        # Estimate time to budget exhaustion
        time_to_exhaustion = None
        if burn_rate > 0 and error_budget_remaining > 0:
            # Hours remaining at current burn rate
            hours_in_window = slo.window.total_seconds() / 3600
            time_to_exhaustion = (error_budget_remaining / 100) * hours_in_window / burn_rate
        
        # Calculate projected compliance
        projected_compliance = current_value
        if burn_rate > 1:
            # We're burning budget faster than expected
            deficit = burn_rate - 1
            projected_compliance = max(0, current_value - deficit * 0.01)
        
        # Determine status
        if current_value >= slo.target_value:
            status = SLOStatus.HEALTHY
        elif current_value >= slo.get_at_risk_value():
            status = SLOStatus.AT_RISK
        elif current_value >= slo.get_breaching_value():
            status = SLOStatus.BREACHING
        else:
            status = SLOStatus.BREACHING
        
        # Check for active breaches
        if error_budget_remaining <= 0:
            status = SLOStatus.BREACHING
        
        compliance = SLOCompliance(
            slo_id=slo_id,
            current_value=current_value,
            target_value=slo.target_value,
            status=status,
            window_start=window_start,
            window_end=window_end,
            total_requests=total_requests,
            good_requests=good_requests,
            bad_requests=bad_requests,
            error_budget_remaining=error_budget_remaining,
            error_budget_used=error_budget_used,
            burn_rate=burn_rate,
            time_to_budget_exhaustion=time_to_exhaustion,
            projected_compliance=projected_compliance,
        )
        
        # Add historical data
        if slo_id in self.compliance:
            compliance.historical_compliance = self.compliance[slo_id].historical_compliance.copy()
        
        compliance.historical_compliance.append(current_value)
        if len(compliance.historical_compliance) > 100:
            compliance.historical_compliance = compliance.historical_compliance[-100:]
        
        self.compliance[slo_id] = compliance
        
        # Record history
        self.history[slo_id].append(SLOHistoryEntry(
            timestamp=datetime.utcnow(),
            compliance=current_value,
            request_count=total_requests,
            error_count=bad_requests,
            p95_latency=metrics.get("p95_latency", 0),
        ))
        
        return compliance
    
    def evaluate_all_slos(self) -> Dict[str, SLOCompliance]:
        """Evaluate all registered SLOs"""
        for slo_id in self.slos:
            self.evaluate_slo(slo_id)
        return self.compliance
    
    def get_slo_status(self, slo_id: str) -> SLOStatus:
        """Get current status of an SLO"""
        if slo_id in self.compliance:
            return self.compliance[slo_id].status
        return SLOStatus.UNKNOWN
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_slos": len(self.slos),
            "healthy_slos": 0,
            "at_risk_slos": 0,
            "breaching_slos": 0,
            "slo_details": [],
            "error_budget_summary": {
                "total_budget_remaining": 0,
                "critical_budgets": [],
            },
        }
        
        for slo_id, slo in self.slos.items():
            compliance = self.evaluate_slo(slo_id)
            
            if compliance:
                if compliance.status == SLOStatus.HEALTHY:
                    report["healthy_slos"] += 1
                elif compliance.status == SLOStatus.AT_RISK:
                    report["at_risk_slos"] += 1
                elif compliance.status == SLOStatus.BREACHING:
                    report["breaching_slos"] += 1
                
                report["slo_details"].append(compliance.to_dict())
                report["error_budget_summary"]["total_budget_remaining"] += compliance.error_budget_remaining
                
                if compliance.error_budget_remaining < 20:
                    report["error_budget_summary"]["critical_budgets"].append({
                        "slo_id": slo_id,
                        "budget_remaining": compliance.error_budget_remaining,
                        "burn_rate": compliance.burn_rate,
                    })
        
        return report
    
    def get_budget_burning_alerts(self) -> List[Dict[str, Any]]:
        """Get alerts for SLOs with critical burn rates"""
        alerts = []
        
        for slo_id, compliance in self.compliance.items():
            slo = self.slos.get(slo_id)
            if not slo:
                continue
            
            # Fast burn alert (>1x burn rate)
            if compliance.burn_rate > slo.burn_rate_threshold * 2:
                alerts.append({
                    "slo_id": slo_id,
                    "slo_name": slo.name,
                    "alert_type": "fast_burn",
                    "severity": "critical",
                    "message": f"Error budget burning at {compliance.burn_rate:.1f}x expected rate",
                    "burn_rate": compliance.burn_rate,
                    "budget_remaining": compliance.error_budget_remaining,
                    "time_to_exhaustion_hours": compliance.time_to_budget_exhaustion,
                })
            
            # Exhaustion warning
            if compliance.time_to_budget_exhaustion and compliance.time_to_budget_exhaustion < 24:
                alerts.append({
                    "slo_id": slo_id,
                    "slo_name": slo.name,
                    "alert_type": "budget_exhaustion",
                    "severity": "warning",
                    "message": f"Error budget will be exhausted in {compliance.time_to_budget_exhaustion:.1f} hours",
                    "time_to_exhaustion_hours": compliance.time_to_budget_exhaustion,
                })
            
            # Breach warning
            if compliance.status == SLOStatus.BREACHING:
                alerts.append({
                    "slo_id": slo_id,
                    "slo_name": slo.name,
                    "alert_type": "slo_breach",
                    "severity": "critical",
                    "message": f"SLO currently breaching: {compliance.compliance_percent:.2f}% vs {slo.target_to_percent():.2f}% target",
                    "current_compliance": compliance.compliance_percent,
                    "target_compliance": slo.target_to_percent(),
                })
        
        return alerts
    
    def get_historical_data(
        self,
        slo_id: str,
        hours: int = 24,
    ) -> List[SLOHistoryEntry]:
        """Get historical data for an SLO"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            entry for entry in self.history.get(slo_id, [])
            if entry.timestamp >= cutoff
        ]


# Convenience functions
def create_slo_tracker() -> SLOTracker:
    """Create a new SLO tracker"""
    return SLOTracker()


def create_default_slos(service_name: str) -> List[SLOTarget]:
    """Create default SLOs for a service"""
    return [
        SLOTarget(
            slo_id=f"{service_name}-availability",
            name=f"{service_name} Availability",
            description=f"99.9% availability target for {service_name}",
            service_name=service_name,
            sli_type=SLIType.AVAILABILITY,
            target_value=0.999,
            window=timedelta(days=30),
            error_budget_percent=0.1,
        ),
        SLOTarget(
            slo_id=f"{service_name}-latency-p95",
            name=f"{service_name} P95 Latency",
            description=f"P95 latency under 500ms for {service_name}",
            service_name=service_name,
            sli_type=SLIType.LATENCY,
            target_value=500.0,  # 500ms
            window=timedelta(days=7),
            at_risk_threshold=0.9,
            breaching_threshold=0.8,
        ),
        SLOTarget(
            slo_id=f"{service_name}-latency-p99",
            name=f"{service_name} P99 Latency",
            description=f"P99 latency under 1000ms for {service_name}",
            service_name=service_name,
            sli_type=SLIType.LATENCY,
            target_value=1000.0,  # 1000ms
            window=timedelta(days=7),
            at_risk_threshold=0.9,
            breaching_threshold=0.8,
        ),
    ]
