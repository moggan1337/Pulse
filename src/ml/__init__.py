"""ML-based Anomaly Detection Module"""

import asyncio
import json
import logging
import math
import pickle
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Optional ML dependencies
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class AnomalyScore:
    """Score representing how anomalous a data point is"""
    value: float  # Raw anomaly score
    normalized: float  # 0-1 scale where 1 is most anomalous
    is_anomaly: bool
    confidence: float  # Confidence in the prediction
    features: Dict[str, float]  # Contributing features
    timestamp: datetime = field(default_factory=datetime.utcnow)
    service_name: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class ModelMetrics:
    """Metrics for model performance tracking"""
    model_name: str
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update(self, actual: bool, predicted: bool) -> None:
        """Update metrics with a new prediction"""
        if actual and predicted:
            self.true_positives += 1
        elif not actual and predicted:
            self.false_positives += 1
        elif actual and not predicted:
            self.false_negatives += 1
        else:
            self.true_negatives += 1
        
        # Recalculate metrics
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives
        tn = self.true_negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        
        self.precision = precision
        self.recall = recall
        self.f1_score = f1
        self.accuracy = accuracy
        self.last_updated = datetime.utcnow()


class FeatureExtractor:
    """
    Extracts features from traces and spans for ML model input.
    
    Features include:
    - Latency metrics (mean, std, min, max, percentiles)
    - Error rates
    - Request rates
    - Service dependency metrics
    - Time-based patterns
    """
    
    def __init__(self):
        self.feature_cache: Dict[str, deque] = {}
        self.cache_size = 1000
    
    def extract_span_features(self, span_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from a single span"""
        features = []
        
        # Duration features
        duration = span_data.get("duration_ms", 0)
        features.append(np.log1p(duration))  # Log-transformed duration
        
        # Error features
        features.append(1.0 if span_data.get("error_flag") else 0.0)
        
        # Attribute-based features
        attributes = span_data.get("attributes", {})
        features.append(float(attributes.get("http.status_code", 200)) / 600)
        
        # Event count
        event_count = len(span_data.get("events", []))
        features.append(np.log1p(event_count))
        
        # Span kind encoding
        kind_map = {"internal": 0, "server": 1, "client": 2, "producer": 3, "consumer": 4}
        features.append(kind_map.get(span_data.get("kind", "internal"), 0) / 4)
        
        return np.array(features)
    
    def extract_trace_features(
        self,
        trace_data: Dict[str, Any],
        historical_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """
        Extract comprehensive features from a trace.
        
        Features:
        - Trace duration statistics
        - Span count and composition
        - Error rate
        - Service diversity
        - Historical deviation scores
        """
        features = []
        spans = trace_data.get("spans", [])
        
        if not spans:
            return np.zeros(20)
        
        # Duration statistics
        durations = [s.get("duration_ms", 0) for s in spans]
        features.extend(self._compute_duration_features(durations, historical_stats))
        
        # Span composition
        features.append(len(spans))
        features.append(np.log1p(len(spans)))
        
        # Error rate
        error_count = sum(1 for s in spans if s.get("error_flag"))
        features.append(error_count / len(spans))
        
        # Service diversity (unique services)
        services = set(s.get("service_name", "unknown") for s in spans)
        features.append(len(services))
        
        # Service-level features
        service_durations = {}
        for span in spans:
            service = span.get("service_name", "unknown")
            if service not in service_durations:
                service_durations[service] = []
            service_durations[service].append(span.get("duration_ms", 0))
        
        # Average services involved
        if service_durations:
            avg_service_durations = [np.mean(d) for d in service_durations.values()]
            features.append(np.mean(avg_service_durations))
            features.append(np.std(avg_service_durations) if len(avg_service_durations) > 1 else 0)
        else:
            features.extend([0, 0])
        
        # Event features
        total_events = sum(len(s.get("events", [])) for s in spans)
        features.append(total_events)
        features.append(np.log1p(total_events))
        
        # Link features (distributed context)
        total_links = sum(len(s.get("links", [])) for s in spans)
        features.append(total_links)
        
        # Status code distribution
        status_codes = [s.get("attributes", {}).get("http.status_code", 200) for s in spans]
        error_rate_status = sum(1 for c in status_codes if c >= 400) / len(status_codes)
        features.append(error_rate_status)
        
        # Parent-child ratio (tree depth indicator)
        parent_count = sum(1 for s in spans if s.get("parent_span_id"))
        features.append(parent_count / len(spans) if spans else 0)
        
        # Historical deviation (requires historical_stats)
        if historical_stats:
            features.append(self._compute_historical_deviation(
                durations, 
                trace_data.get("service_name"), 
                historical_stats
            ))
        else:
            features.append(0)
        
        return np.array(features)
    
    def _compute_duration_features(
        self,
        durations: List[float],
        historical_stats: Optional[Dict[str, Tuple[float, float]]],
    ) -> List[float]:
        """Compute duration-based statistical features"""
        features = []
        
        if not durations:
            return [0] * 6
        
        durations_arr = np.array(durations)
        
        # Basic statistics
        features.append(np.mean(durations_arr))
        features.append(np.std(durations_arr))
        features.append(np.min(durations_arr))
        features.append(np.max(durations_arr))
        
        # Percentiles
        features.append(np.percentile(durations_arr, 50))
        features.append(np.percentile(durations_arr, 95))
        
        return features
    
    def _compute_historical_deviation(
        self,
        durations: List[float],
        service_name: Optional[str],
        historical_stats: Dict[str, Tuple[float, float]],
    ) -> float:
        """Compute how much current durations deviate from historical baseline"""
        if not service_name or service_name not in historical_stats:
            return 0.0
        
        mean, std = historical_stats[service_name]
        if std == 0:
            return 0.0
        
        current_mean = np.mean(durations)
        return abs(current_mean - mean) / std
    
    def extract_time_series_features(
        self,
        values: List[float],
        window_size: int = 10,
    ) -> np.ndarray:
        """Extract features from a time series for LSTM input"""
        if len(values) < window_size:
            # Pad with zeros if not enough data
            values = values + [0] * (window_size - len(values))
        
        window = values[-window_size:]
        arr = np.array(window)
        
        features = []
        
        # Normalized values
        features.extend(arr / (np.max(arr) + 1e-8))
        
        # Differences (velocity)
        diffs = np.diff(arr)
        features.extend([0] + list(diffs / (np.abs(arr[:-1]) + 1e-8)))
        
        # Rolling statistics
        features.append(np.mean(arr))
        features.append(np.std(arr))
        features.append(np.max(arr) - np.min(arr))
        
        return np.array(features)


class IsolationForestDetector:
    """
    Anomaly detection using Isolation Forest algorithm.
    
    Isolation Forest isolates anomalies by randomly selecting features
    and split values. Anomalies are easier to isolate and have shorter
    path lengths in the tree structure.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: int = 256,
        random_state: int = 42,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for IsolationForestDetector")
        
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.is_fitted = False
        self.training_data: List[np.ndarray] = []
        self.metrics = ModelMetrics(model_name="IsolationForest")
    
    def train(self, traces: List[Dict[str, Any]], labels: Optional[List[bool]] = None) -> None:
        """
        Train the Isolation Forest model.
        
        Args:
            traces: List of trace dictionaries
            labels: Optional ground truth labels for evaluation
        """
        logger.info(f"Training Isolation Forest on {len(traces)} traces")
        
        # Extract features
        features_list = []
        for trace in traces:
            features = self.feature_extractor.extract_trace_features(trace)
            features_list.append(features)
            self.training_data.append(features)
        
        if not features_list:
            logger.warning("No training data available")
            return
        
        X = np.array(features_list)
        
        # Handle any NaN or Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        # Update metrics if labels provided
        if labels:
            predictions = self.predict_batch(traces)
            for actual, predicted in zip(labels, predictions):
                self.metrics.update(actual, predicted.is_anomaly)
        
        logger.info("Isolation Forest training complete")
    
    def predict(self, trace: Dict[str, Any]) -> AnomalyScore:
        """
        Predict if a trace is anomalous.
        
        Returns an AnomalyScore with details about the prediction.
        """
        if not self.is_fitted:
            # Return default score if not trained
            return AnomalyScore(
                value=0.0,
                normalized=0.0,
                is_anomaly=False,
                confidence=0.0,
                features={},
            )
        
        # Extract features
        features = self.feature_extractor.extract_trace_features(trace)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(features.reshape(1, -1))
        
        # Get anomaly score (negative means more anomalous)
        raw_score = self.model.score_samples(X)[0]
        
        # Convert to 0-1 scale where 1 is most anomalous
        normalized_score = self._normalize_score(raw_score)
        
        # Decision function
        is_anomaly = self.model.predict(X)[0] == -1
        
        # Confidence based on score magnitude
        confidence = min(abs(raw_score) / 2, 1.0)
        
        return AnomalyScore(
            value=raw_score,
            normalized=normalized_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            features=self._get_feature_importance(features),
            trace_id=trace.get("trace_id"),
        )
    
    def predict_batch(self, traces: List[Dict[str, Any]]) -> List[AnomalyScore]:
        """Predict anomalies for multiple traces"""
        return [self.predict(trace) for trace in traces]
    
    def _normalize_score(self, raw_score: float) -> float:
        """Normalize raw anomaly score to 0-1 range"""
        # Isolation Forest scores are typically in range [-0.5, 0.5]
        # Lower scores indicate more anomalous behavior
        normalized = (0.5 - raw_score) / 1.0
        return max(0.0, min(1.0, normalized))
    
    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance based on deviation from mean"""
        feature_names = [
            "mean_duration", "std_duration", "min_duration", "max_duration",
            "p50_duration", "p95_duration", "span_count", "log_span_count",
            "error_rate", "service_diversity", "avg_service_duration",
            "std_service_duration", "total_events", "log_events", "total_links",
            "error_rate_status", "parent_ratio", "historical_deviation",
        ]
        
        # Approximate feature importance using feature magnitude
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(features):
                importance[name] = float(abs(features[i]))
        
        return importance
    
    def partial_fit(self, trace: Dict[str, Any], is_anomaly: bool = None) -> None:
        """
        Incrementally update the model with new data.
        
        This is useful for online learning scenarios.
        """
        features = self.feature_extractor.extract_trace_features(trace)
        self.training_data.append(features)
        
        # Keep only recent data points
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-5000:]
        
        # Retrain periodically (every 100 new samples)
        if len(self.training_data) % 100 == 0:
            X = np.array(self.training_data[-1000:])  # Use recent samples
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled)
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "is_fitted": self.is_fitted,
                "contamination": self.contamination,
            }, f)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_fitted = data["is_fitted"]
        self.contamination = data["contamination"]
        logger.info(f"Model loaded from {path}")


class LSTMANomalyDetector:
    """
    LSTM-based anomaly detection for time series trace data.
    
    Uses a Long Short-Term Memory network to learn normal patterns
    in trace behavior and detect deviations.
    """
    
    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 10,
        threshold: float = 0.05,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTMANomalyDetector")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.threshold = threshold
        
        # Build model
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.feature_extractor = FeatureExtractor()
        self.sequence_buffer: deque = deque(maxlen=1000)
        self.is_trained = False
        self.metrics = ModelMetrics(model_name="LSTM")
    
    def _build_model(self) -> nn.Module:
        """Build the LSTM model architecture"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        class LSTMAnomaly(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2,
                )
                self.fc = nn.Linear(hidden_size, input_size)
            
            def forward(self, x):
                # x shape: (batch, seq_len, input_size)
                lstm_out, _ = self.lstm(x)
                # Use last output
                output = self.fc(lstm_out[:, -1, :])
                return output
        
        model = LSTMAnomaly(self.input_size, self.hidden_size, self.num_layers)
        return model.to(device)
    
    def _prepare_sequence(
        self,
        current_trace: Dict[str, Any],
    ) -> torch.Tensor:
        """Prepare input sequence for LSTM"""
        # Get features from current trace
        current_features = self.feature_extractor.extract_trace_features(current_trace)
        
        # Add to buffer
        self.sequence_buffer.append(current_features)
        
        # Build sequence
        if len(self.sequence_buffer) < self.sequence_length:
            # Pad with zeros
            padding = [np.zeros(self.input_size) for _ in range(self.sequence_length - len(self.sequence_buffer))]
            sequence = padding + list(self.sequence_buffer)
        else:
            sequence = list(self.sequence_buffer)[-self.sequence_length:]
        
        # Convert to tensor
        sequence_arr = np.array(sequence)
        
        # Normalize
        mean = sequence_arr.mean(axis=0)
        std = sequence_arr.std(axis=0) + 1e-8
        normalized = (sequence_arr - mean) / std
        
        tensor = torch.FloatTensor(normalized).unsqueeze(0)  # Add batch dimension
        
        device = next(self.model.parameters()).device
        return tensor.to(device)
    
    def train(
        self,
        traces: List[Dict[str, Any]],
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """Train the LSTM model on normal trace data"""
        logger.info(f"Training LSTM on {len(traces)} traces for {epochs} epochs")
        
        self.model.train()
        device = next(self.model.parameters()).device
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Prepare training sequences
            sequences = []
            for trace in traces:
                try:
                    features = self.feature_extractor.extract_trace_features(trace)
                    self.sequence_buffer.append(features)
                    sequences.append(features)
                except Exception as e:
                    logger.warning(f"Failed to extract features: {e}")
                    continue
            
            if len(sequences) < batch_size:
                logger.warning("Not enough sequences for training")
                return
            
            # Create batches
            for i in range(0, len(sequences) - self.sequence_length, batch_size):
                batch_sequences = sequences[i:i + batch_size]
                
                # Prepare input sequences
                batch_tensors = []
                for seq_features in batch_sequences[-self.sequence_length:]:
                    batch_tensors.append(seq_features)
                
                if len(batch_tensors) < self.sequence_length:
                    continue
                
                # Stack and normalize
                batch_arr = np.array(batch_tensors[-self.sequence_length:])
                mean = batch_arr.mean(axis=0)
                std = batch_arr.std(axis=0) + 1e-8
                normalized = (batch_arr - mean) / std
                
                X = torch.FloatTensor(normalized).unsqueeze(0).to(device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(X)
                
                # Reconstruction loss
                loss = self.criterion(output, X[:, -1, :])
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info("LSTM training complete")
    
    def predict(self, trace: Dict[str, Any]) -> AnomalyScore:
        """Predict if a trace is anomalous"""
        if not self.is_trained:
            return AnomalyScore(
                value=0.0,
                normalized=0.0,
                is_anomaly=False,
                confidence=0.0,
                features={},
                trace_id=trace.get("trace_id"),
            )
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        try:
            # Get current features
            current_features = self.feature_extractor.extract_trace_features(trace)
            
            # Add to buffer
            self.sequence_buffer.append(current_features)
            
            # Prepare input
            if len(self.sequence_buffer) < self.sequence_length:
                padding = [np.zeros(self.input_size) for _ in range(self.sequence_length - len(self.sequence_buffer))]
                sequence = padding + list(self.sequence_buffer)
            else:
                sequence = list(self.sequence_buffer)[-self.sequence_length:]
            
            sequence_arr = np.array(sequence)
            mean = sequence_arr.mean(axis=0)
            std = sequence_arr.std(axis=0) + 1e-8
            normalized = (sequence_arr - mean) / std
            
            X = torch.FloatTensor(normalized).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                output = self.model(X)
                reconstruction_error = self.criterion(output, X[:, -1, :]).item()
            
            # Determine if anomalous
            is_anomaly = reconstruction_error > self.threshold
            
            # Normalize score
            normalized_score = min(reconstruction_error / (self.threshold * 2), 1.0)
            
            return AnomalyScore(
                value=reconstruction_error,
                normalized=normalized_score,
                is_anomaly=is_anomaly,
                confidence=min(normalized_score * 2, 1.0),
                features={"reconstruction_error": reconstruction_error},
                trace_id=trace.get("trace_id"),
            )
        
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return AnomalyScore(
                value=0.0,
                normalized=0.0,
                is_anomaly=False,
                confidence=0.0,
                features={},
                trace_id=trace.get("trace_id"),
            )
    
    def update_threshold(self, new_threshold: float) -> None:
        """Update the anomaly threshold"""
        self.threshold = new_threshold
        logger.info(f"LSTM threshold updated to {new_threshold}")
    
    def save(self, path: str) -> None:
        """Save model state"""
        device = torch.device("cpu")
        self.model.to(device)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "threshold": self.threshold,
            "is_trained": self.is_trained,
        }, path)
        logger.info(f"LSTM model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model state"""
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.threshold = checkpoint["threshold"]
        self.is_trained = checkpoint["is_trained"]
        logger.info(f"LSTM model loaded from {path}")


class EnsembleDetector:
    """
    Ensemble anomaly detector combining multiple models.
    
    Combines Isolation Forest and LSTM predictions for more robust detection.
    Uses weighted voting based on model confidence.
    """
    
    def __init__(
        self,
        if_weight: float = 0.5,
        lstm_weight: float = 0.5,
        anomaly_threshold: float = 0.7,
    ):
        self.if_detector: Optional[IsolationForestDetector] = None
        self.lstm_detector: Optional[LSTMANomalyDetector] = None
        
        self.if_weight = if_weight
        self.lstm_weight = lstm_weight
        self.anomaly_threshold = anomaly_threshold
        
        self.ensemble_metrics = ModelMetrics(model_name="Ensemble")
    
    def initialize_models(self) -> None:
        """Initialize all available models"""
        if SKLEARN_AVAILABLE:
            self.if_detector = IsolationForestDetector()
            logger.info("Isolation Forest detector initialized")
        
        if TORCH_AVAILABLE:
            self.lstm_detector = LSTMANomalyDetector()
            logger.info("LSTM detector initialized")
        
        if not self.if_detector and not self.lstm_detector:
            logger.warning("No ML models available. Install scikit-learn and/or PyTorch.")
    
    def train(
        self,
        traces: List[Dict[str, Any]],
        labels: Optional[List[bool]] = None,
        epochs: int = 10,
    ) -> None:
        """Train all available models"""
        if self.if_detector:
            logger.info("Training Isolation Forest...")
            self.if_detector.train(traces, labels)
        
        if self.lstm_detector:
            logger.info("Training LSTM...")
            self.lstm_detector.train(traces, epochs=epochs)
    
    def predict(self, trace: Dict[str, Any]) -> AnomalyScore:
        """
        Predict anomaly using ensemble of models.
        
        Combines predictions from all available models with weighted averaging.
        """
        scores = []
        weights = []
        
        if self.if_detector and self.if_detector.is_fitted:
            score = self.if_detector.predict(trace)
            scores.append(score)
            weights.append(self.if_weight * score.confidence)
        
        if self.lstm_detector and self.lstm_detector.is_trained:
            score = self.lstm_detector.predict(trace)
            scores.append(score)
            weights.append(self.lstm_weight * score.confidence)
        
        if not scores:
            return AnomalyScore(
                value=0.0,
                normalized=0.0,
                is_anomaly=False,
                confidence=0.0,
                features={},
                trace_id=trace.get("trace_id"),
            )
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Weighted average of scores
        ensemble_score = sum(s.normalized * w for s, w in zip(scores, weights))
        
        # Determine if anomalous
        is_anomaly = ensemble_score >= self.anomaly_threshold
        
        # Combine features
        combined_features = {}
        for score in scores:
            combined_features.update(score.features)
        
        return AnomalyScore(
            value=ensemble_score,
            normalized=ensemble_score,
            is_anomaly=is_anomaly,
            confidence=sum(s.confidence * w for s, w in zip(scores, weights)),
            features=combined_features,
            trace_id=trace.get("trace_id"),
        )
    
    def predict_batch(self, traces: List[Dict[str, Any]]) -> List[AnomalyScore]:
        """Predict anomalies for multiple traces"""
        return [self.predict(trace) for trace in traces]


class AnomalyDetector:
    """
    Main anomaly detection interface with model management.
    
    Provides a unified API for training, predicting, and managing
    multiple anomaly detection models.
    """
    
    def __init__(
        self,
        detection_threshold: float = 0.7,
        enable_iforest: bool = True,
        enable_lstm: bool = True,
    ):
        self.detection_threshold = detection_threshold
        
        # Initialize ensemble
        self.ensemble = EnsembleDetector(
            if_weight=0.6 if enable_iforest else 0,
            lstm_weight=0.4 if enable_lstm else 0,
            anomaly_threshold=detection_threshold,
        )
        
        # Initialize individual models if enabled
        if enable_iforest and SKLEARN_AVAILABLE:
            self.ensemble.if_detector = IsolationForestDetector()
        
        if enable_lstm and TORCH_AVAILABLE:
            self.ensemble.lstm_detector = LSTMANomalyDetector()
        
        self.feature_extractor = FeatureExtractor()
        self.is_initialized = False
        
        # Statistics
        self.stats = {
            "total_predictions": 0,
            "anomalies_detected": 0,
            "models_trained": 0,
        }
    
    def initialize(self) -> None:
        """Initialize the detector"""
        self.ensemble.initialize_models()
        self.is_initialized = True
        logger.info("Anomaly detector initialized")
    
    def train(
        self,
        traces: List[Dict[str, Any]],
        labels: Optional[List[bool]] = None,
        epochs: int = 10,
    ) -> Dict[str, ModelMetrics]:
        """
        Train all available models on the provided data.
        
        Args:
            traces: List of trace dictionaries
            labels: Optional ground truth labels
            epochs: Number of training epochs (for LSTM)
        
        Returns:
            Dictionary of model metrics
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Training anomaly detectors on {len(traces)} traces")
        
        # Train ensemble
        self.ensemble.train(traces, labels, epochs)
        
        # Update stats
        if self.ensemble.if_detector and self.ensemble.if_detector.is_fitted:
            self.stats["models_trained"] += 1
        if self.ensemble.lstm_detector and self.ensemble.lstm_detector.is_trained:
            self.stats["models_trained"] += 1
        
        # Return metrics
        metrics = {}
        if self.ensemble.if_detector:
            metrics["isolation_forest"] = self.ensemble.if_detector.metrics
        if self.ensemble.lstm_detector:
            metrics["lstm"] = self.ensemble.lstm_detector.metrics
        metrics["ensemble"] = self.ensemble.ensemble_metrics
        
        return metrics
    
    def detect(
        self,
        trace: Dict[str, Any],
        update_model: bool = False,
    ) -> AnomalyScore:
        """
        Detect anomalies in a trace.
        
        Args:
            trace: Trace dictionary
            update_model: Whether to update models online
        
        Returns:
            AnomalyScore with detection results
        """
        # Make prediction
        score = self.ensemble.predict(trace)
        
        # Update statistics
        self.stats["total_predictions"] += 1
        if score.is_anomaly:
            self.stats["anomalies_detected"] += 1
        
        # Online learning
        if update_model and self.ensemble.if_detector:
            self.ensemble.if_detector.partial_fit(trace, score.is_anomaly)
        
        return score
    
    def detect_batch(
        self,
        traces: List[Dict[str, Any]],
    ) -> List[AnomalyScore]:
        """Detect anomalies in multiple traces"""
        return [self.detect(trace) for trace in traces]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            **self.stats,
            "anomaly_rate": (
                self.stats["anomalies_detected"] / max(1, self.stats["total_predictions"])
            ),
            "models_available": {
                "isolation_forest": self.ensemble.if_detector is not None,
                "lstm": self.ensemble.lstm_detector is not None,
            },
        }
    
    def save_models(self, directory: str) -> None:
        """Save all models to disk"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        if self.ensemble.if_detector and self.ensemble.if_detector.is_fitted:
            self.ensemble.if_detector.save(f"{directory}/isolation_forest.pkl")
        
        if self.ensemble.lstm_detector and self.ensemble.lstm_detector.is_trained:
            self.ensemble.lstm_detector.save(f"{directory}/lstm.pt")
    
    def load_models(self, directory: str) -> None:
        """Load all models from disk"""
        import os
        
        if self.ensemble.if_detector:
            if_path = f"{directory}/isolation_forest.pkl"
            if os.path.exists(if_path):
                self.ensemble.if_detector.load(if_path)
        
        if self.ensemble.lstm_detector:
            lstm_path = f"{directory}/lstm.pt"
            if os.path.exists(lstm_path):
                self.ensemble.lstm_detector.load(lstm_path)


# Utility functions
def create_detector(
    enable_iforest: bool = True,
    enable_lstm: bool = True,
    threshold: float = 0.7,
) -> AnomalyDetector:
    """Create a configured anomaly detector"""
    detector = AnomalyDetector(
        enable_iforest=enable_iforest,
        enable_lstm=enable_lstm,
        detection_threshold=threshold,
    )
    detector.initialize()
    return detector


def generate_synthetic_traces(
    num_traces: int = 100,
    anomaly_rate: float = 0.1,
    services: List[str] = None,
) -> Tuple[List[Dict[str, Any]], List[bool]]:
    """
    Generate synthetic trace data for testing.
    
    Returns traces with ground truth labels.
    """
    if services is None:
        services = ["api-gateway", "user-service", "order-service", "payment-service"]
    
    traces = []
    labels = []
    
    for i in range(num_traces):
        # Decide if this should be anomalous
        is_anomaly = np.random.random() < anomaly_rate
        
        num_spans = np.random.randint(5, 20)
        spans = []
        
        for j in range(num_spans):
            # Generate duration based on anomaly status
            if is_anomaly:
                # Anomalous traces have extreme values
                duration = np.random.exponential(5000) if np.random.random() < 0.3 else np.random.exponential(50)
            else:
                duration = np.random.exponential(100)
            
            span = {
                "name": f"{services[j % len(services)]}.{np.random.choice(['handler', 'process', 'validate'])}",
                "trace_id": f"trace-{i:06d}",
                "span_id": f"span-{i:06d}-{j:03d}",
                "parent_span_id": f"span-{i:06d}-{j-1:03d}" if j > 0 else None,
                "service_name": services[j % len(services)],
                "duration_ms": duration,
                "error_flag": is_anomaly and np.random.random() < 0.5,
                "attributes": {
                    "http.status_code": 500 if is_anomaly and np.random.random() < 0.3 else 200,
                },
                "events": [{"name": "processing"}] if np.random.random() < 0.3 else [],
            }
            spans.append(span)
        
        trace = {
            "trace_id": f"trace-{i:06d}",
            "spans": spans,
            "start_time": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
            "end_time": (datetime.utcnow() - timedelta(minutes=i) + timedelta(milliseconds=sum(s["duration_ms"] for s in spans))).isoformat(),
            "total_duration_ms": sum(s["duration_ms"] for s in spans),
        }
        
        traces.append(trace)
        labels.append(is_anomaly)
    
    return traces, labels
