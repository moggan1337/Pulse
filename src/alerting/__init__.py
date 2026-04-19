"""Predictive Alerting Module"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status values"""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class AlertType(Enum):
    """Types of alerts"""
    ANOMALY = "anomaly"
    THRESHOLD = "threshold"
    PREDICTIVE = "predictive"
    SLO_BREACH = "slo_breach"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    AVAILABILITY = "availability"


@dataclass
class Alert:
    """Represents an alert"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    
    # Targets
    service_name: Optional[str] = None
    metric_name: Optional[str] = None
    
    # Values at alert time
    current_value: float = 0.0
    threshold_value: float = 0.0
    
    # Time
    fired_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    
    # Status
    status: AlertStatus = AlertStatus.FIRING
    
    # Context
    trace_ids: List[str] = field(default_factory=list)
    anomaly_score: Optional[float] = None
    prediction: Optional[Dict[str, float]] = None
    
    # Actions
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def duration_seconds(self) -> float:
        """Calculate alert duration"""
        end = self.resolved_at or datetime.utcnow()
        return (end - self.fired_at).total_seconds()
    
    def acknowledge(self) -> None:
        """Acknowledge the alert"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
    
    def resolve(self) -> None:
        """Resolve the alert"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
    
    def suppress(self, duration_seconds: int = 3600) -> None:
        """Suppress the alert"""
        self.status = AlertStatus.SUPPRESSED
        self.annotations["suppressed_until"] = (
            datetime.utcnow() + timedelta(seconds=duration_seconds)
        ).isoformat()


@dataclass
class AlertRule:
    """Defines an alerting rule"""
    rule_id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    
    # Conditions
    service_name: Optional[str] = None
    metric_name: str
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    
    # Time constraints
    evaluation_interval: int = 60  # seconds
    window_size: int = 300  # seconds
    minimum_occurrences: int = 1
    
    # Actions
    enabled: bool = True
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Suppression
    cooldown_seconds: int = 300
    suppress_duration_seconds: int = 0
    
    def evaluate(self, value: float, occurrences: int) -> bool:
        """Evaluate if the rule should fire"""
        if not self.enabled:
            return False
        
        # Check condition
        triggered = False
        if self.condition == "gt":
            triggered = value > self.threshold
        elif self.condition == "gte":
            triggered = value >= self.threshold
        elif self.condition == "lt":
            triggered = value < self.threshold
        elif self.condition == "lte":
            triggered = value <= self.threshold
        elif self.condition == "eq":
            triggered = value == self.threshold
        
        return triggered and occurrences >= self.minimum_occurrences


@dataclass
class Prediction:
    """Represents a prediction for forecasting"""
    metric_name: str
    service_name: Optional[str]
    
    # Predicted values
    predicted_value: float
    predicted_at: datetime
    prediction_horizon: timedelta  # How far in the future
    
    # Confidence
    confidence: float  # 0-1
    
    # Predicted timeframe
    predicted_time: datetime
    
    # Additional context
    model_name: str = "moving_average"
    features: Dict[str, float] = field(default_factory=dict)


class TimeSeriesForecaster:
    """
    Simple time series forecasting for metrics prediction.
    
    Uses multiple methods:
    - Moving average
    - Exponential smoothing
    - Trend extrapolation
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def add_observation(
        self,
        metric_name: str,
        service_name: Optional[str],
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add a new observation to the time series"""
        key = self._get_key(metric_name, service_name)
        ts = timestamp or datetime.utcnow()
        self.history[key].append((ts, value))
    
    def predict(
        self,
        metric_name: str,
        service_name: Optional[str],
        horizon_seconds: int = 60,
    ) -> Prediction:
        """Predict future value of a metric"""
        key = self._get_key(metric_name, service_name)
        series = self.history.get(key, deque())
        
        if len(series) < 2:
            return Prediction(
                metric_name=metric_name,
                service_name=service_name,
                predicted_value=0.0,
                predicted_at=datetime.utcnow(),
                prediction_horizon=timedelta(seconds=horizon_seconds),
                predicted_time=datetime.utcnow() + timedelta(seconds=horizon_seconds),
                confidence=0.0,
            )
        
        # Extract values
        values = [v for _, v in series]
        times = [t for t, _ in series]
        
        # Calculate prediction using multiple methods
        ma_pred = self._moving_average(values)
        trend_pred = self._trend_extrapolation(values, times, horizon_seconds)
        exp_pred = self._exponential_smoothing(values)
        
        # Combine predictions
        predicted_value = (ma_pred + trend_pred + exp_pred) / 3
        
        # Calculate confidence based on variance
        variance = np.var(values) if len(values) > 1 else 0
        confidence = max(0, 1 - min(variance / 10000, 1))
        
        return Prediction(
            metric_name=metric_name,
            service_name=service_name,
            predicted_value=predicted_value,
            predicted_at=datetime.utcnow(),
            prediction_horizon=timedelta(seconds=horizon_seconds),
            predicted_time=datetime.utcnow() + timedelta(seconds=horizon_seconds),
            confidence=confidence,
            features={
                "moving_average": ma_pred,
                "trend_prediction": trend_pred,
                "exponential_smoothing": exp_pred,
            },
        )
    
    def _moving_average(self, values: List[float]) -> float:
        """Calculate simple moving average"""
        if not values:
            return 0.0
        return np.mean(values[-self.window_size:])
    
    def _trend_extrapolation(
        self,
        values: List[float],
        times: List[datetime],
        horizon_seconds: int,
    ) -> float:
        """Extrapolate based on linear trend"""
        if len(values) < 2:
            return values[-1] if values else 0.0
        
        # Simple linear regression
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        
        # Extrapolate
        future_x = len(values) + horizon_seconds / 60  # Assuming 1 point per minute
        predicted = np.polyval(coeffs, future_x)
        
        return max(0, predicted)
    
    def _exponential_smoothing(self, values: List[float], alpha: float = 0.3) -> float:
        """Apply exponential smoothing"""
        if not values:
            return 0.0
        
        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        return smoothed
    
    def _get_key(self, metric_name: str, service_name: Optional[str]) -> str:
        """Get unique key for metric + service combination"""
        if service_name:
            return f"{service_name}:{metric_name}"
        return metric_name


class ThresholdDetector:
    """
    Detects threshold violations in metrics.
    """
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.violation_counts: Dict[str, int] = defaultdict(int)
        self.last_alert_time: Dict[str, datetime] = {}
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alerting rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> None:
        """Remove an alerting rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
    
    def evaluate_rules(
        self,
        metrics: Dict[str, Dict[str, float]],
    ) -> List[Alert]:
        """
        Evaluate all rules against current metrics.
        
        Args:
            metrics: Dict of service_name -> metric_name -> value
        
        Returns:
            List of firing alerts
        """
        firing_alerts = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Get metric value
            if rule.service_name:
                service_metrics = metrics.get(rule.service_name, {})
            else:
                # Aggregate across all services
                service_metrics = {}
                for sm in metrics.values():
                    for k, v in sm.items():
                        service_metrics[k] = service_metrics.get(k, 0) + v
                if service_metrics.get(rule.metric_name):
                    service_metrics[rule.metric_name] /= len(metrics)
            
            value = service_metrics.get(rule.metric_name, 0)
            
            # Update violation count
            rule_key = f"{rule.rule_id}:{rule.service_name or 'global'}"
            
            if rule.evaluate(value, 1):  # For now, single occurrence
                self.violation_counts[rule_key] += 1
                
                # Check if rule should fire
                if self.violation_counts[rule_key] >= rule.minimum_occurrences:
                    # Check cooldown
                    last_time = self.last_alert_time.get(rule_key)
                    if last_time:
                        elapsed = (datetime.utcnow() - last_time).total_seconds()
                        if elapsed < rule.cooldown_seconds:
                            continue
                    
                    # Fire alert
                    alert = self._create_alert(rule, value)
                    firing_alerts.append(alert)
                    self.last_alert_time[rule_key] = datetime.utcnow()
            else:
                self.violation_counts[rule_key] = 0
        
        return firing_alerts
    
    def _create_alert(self, rule: AlertRule, current_value: float) -> Alert:
        """Create an alert from a rule"""
        return Alert(
            alert_id=f"alert-{rule.rule_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=f"{rule.name}: {current_value:.2f} {rule.condition} {rule.threshold}",
            description=rule.description,
            service_name=rule.service_name,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold_value=rule.threshold,
            labels=rule.labels,
            annotations=rule.annotations,
        )


class PredictiveAlerter:
    """
    Predictive alerting system that forecasts issues before they occur.
    
    Combines:
    - Real-time anomaly detection
    - Time series forecasting
    - Threshold-based alerting
    - SLO tracking
    """
    
    def __init__(
        self,
        forecast_horizon_seconds: int = 300,
        prediction_threshold: float = 0.8,
    ):
        self.forecast_horizon = forecast_horizon_seconds
        self.prediction_threshold = prediction_threshold
        
        # Components
        self.forecaster = TimeSeriesForecaster()
        self.threshold_detector = ThresholdDetector()
        
        # State
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Metrics tracking
        self.current_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Statistics
        self.stats = {
            "total_alerts_fired": 0,
            "alerts_resolved": 0,
            "predictions_made": 0,
            "false_positives": 0,
        }
        
        # Setup default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default alerting rules"""
        default_rules = [
            AlertRule(
                rule_id="error-rate-high",
                name="High Error Rate",
                description="Error rate exceeds 5%",
                alert_type=AlertType.ERROR_RATE,
                severity=AlertSeverity.HIGH,
                metric_name="error_rate",
                condition="gt",
                threshold=0.05,
            ),
            AlertRule(
                rule_id="latency-p95-high",
                name="High P95 Latency",
                description="P95 latency exceeds 1000ms",
                alert_type=AlertType.LATENCY,
                severity=AlertSeverity.MEDIUM,
                metric_name="p95_duration_ms",
                condition="gt",
                threshold=1000,
            ),
            AlertRule(
                rule_id="availability-low",
                name="Low Availability",
                description="Availability drops below 99%",
                alert_type=AlertType.AVAILABILITY,
                severity=AlertSeverity.CRITICAL,
                metric_name="success_rate",
                condition="lt",
                threshold=0.99,
            ),
        ]
        
        for rule in default_rules:
            self.threshold_detector.add_rule(rule)
    
    def update_metrics(
        self,
        metrics: Dict[str, Dict[str, float]],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update current metrics and check for alerts"""
        self.current_metrics = metrics
        ts = timestamp or datetime.utcnow()
        
        # Update forecaster with new observations
        for service_name, service_metrics in metrics.items():
            for metric_name, value in service_metrics.items():
                self.forecaster.add_observation(metric_name, service_name, value, ts)
        
        # Check threshold-based alerts
        alerts = self.threshold_detector.evaluate_rules(metrics)
        
        for alert in alerts:
            self._handle_alert(alert)
    
    def predict_and_alert(
        self,
        metric_name: str,
        service_name: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Make a prediction and create an alert if needed.
        
        Returns an alert if prediction exceeds threshold, None otherwise.
        """
        self.stats["predictions_made"] += 1
        
        # Make prediction
        prediction = self.forecaster.predict(
            metric_name,
            service_name,
            self.forecast_horizon,
        )
        
        # Check if prediction is concerning
        if prediction.confidence < 0.5:
            # Low confidence, don't alert
            return None
        
        # Check if predicted value exceeds threshold
        alert = None
        
        if "error" in metric_name.lower():
            # Predict error rate
            if prediction.predicted_value > 0.05:  # 5% threshold
                alert = self._create_predictive_alert(
                    metric_name,
                    service_name,
                    prediction,
                    AlertSeverity.HIGH,
                    f"Predicted error rate: {prediction.predicted_value*100:.1f}%",
                )
        
        elif "latency" in metric_name.lower():
            # Predict latency
            if prediction.predicted_value > 1000:  # 1000ms threshold
                alert = self._create_predictive_alert(
                    metric_name,
                    service_name,
                    prediction,
                    AlertSeverity.MEDIUM,
                    f"Predicted P95 latency: {prediction.predicted_value:.0f}ms",
                )
        
        if alert:
            self._handle_alert(alert)
        
        return alert
    
    def _create_predictive_alert(
        self,
        metric_name: str,
        service_name: Optional[str],
        prediction: Prediction,
        severity: AlertSeverity,
        description: str,
    ) -> Alert:
        """Create a predictive alert"""
        return Alert(
            alert_id=f"predictive-{metric_name}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            alert_type=AlertType.PREDICTIVE,
            severity=severity,
            title=f"Predicted {metric_name} issue" + (f" for {service_name}" if service_name else ""),
            description=description,
            service_name=service_name,
            metric_name=metric_name,
            current_value=prediction.predicted_value,
            prediction={
                "horizon_seconds": self.forecast_horizon,
                "confidence": prediction.confidence,
                "model": prediction.model_name,
            },
        )
    
    def _handle_alert(self, alert: Alert) -> None:
        """Handle a firing alert"""
        # Check for deduplication
        for existing_alert in self.alerts.values():
            if (
                existing_alert.alert_type == alert.alert_type and
                existing_alert.service_name == alert.service_name and
                existing_alert.status == AlertStatus.FIRING
            ):
                # Don't create duplicate
                return
        
        # Store and notify
        self.alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.stats["total_alerts_fired"] += 1
        
        logger.info(f"Alert fired: {alert.title} ({alert.severity.value})")
        
        # Call callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledge()
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolve()
            self.stats["alerts_resolved"] += 1
            return True
        return False
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        service_name: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
    ) -> List[Alert]:
        """Get active alerts with optional filters"""
        active = [a for a in self.alerts.values() if a.status == AlertStatus.FIRING]
        
        if severity:
            active = [a for a in active if a.severity == severity]
        if service_name:
            active = [a for a in active if a.service_name == service_name]
        if alert_type:
            active = [a for a in active if a.alert_type == alert_type]
        
        return sorted(active, key=lambda a: (a.severity.value, a.fired_at), reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        by_service = defaultdict(int)
        
        for alert in self.alerts.values():
            if alert.status == AlertStatus.FIRING:
                by_severity[alert.severity.value] += 1
                by_type[alert.alert_type.value] += 1
                if alert.service_name:
                    by_service[alert.service_name] += 1
        
        return {
            "total_active": len([a for a in self.alerts.values() if a.status == AlertStatus.FIRING]),
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "by_service": dict(by_service),
            "total_fired": self.stats["total_alerts_fired"],
            "total_resolved": self.stats["alerts_resolved"],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alerter statistics"""
        return {
            **self.stats,
            "active_alerts": len([a for a in self.alerts.values() if a.status == AlertStatus.FIRING]),
            "prediction_accuracy": (
                1 - self.stats["false_positives"] / max(1, self.stats["predictions_made"])
            ),
        }


# Convenience function
def create_alerter(
    forecast_horizon: int = 300,
    threshold: float = 0.8,
) -> PredictiveAlerter:
    """Create a configured predictive alerter"""
    return PredictiveAlerter(
        forecast_horizon_seconds=forecast_horizon,
        prediction_threshold=threshold,
    )
