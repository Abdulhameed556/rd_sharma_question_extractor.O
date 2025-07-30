"""
Fallback handler for RD Sharma Question Extractor.

This module provides backup model management and failover logic for
ensuring service continuity during model outages.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import statistics

from utils.logger import get_logger
from utils.exceptions import LLMInterfaceError
from config import config

logger = get_logger(__name__)


class ModelStatus(Enum):
    """Model status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class ModelHealth:
    """Model health information."""
    model_name: str
    status: ModelStatus
    response_time: float
    error_rate: float
    last_check: datetime
    consecutive_failures: int
    total_requests: int
    successful_requests: int


class FallbackHandler:
    """Handles model failover and backup management."""

    def __init__(self, config):
        """Initialize fallback handler."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Model configuration
        self.primary_model = config.groq_model
        self.backup_models = self._get_backup_models()
        self.current_model = self.primary_model
        
        # Health tracking
        self.model_health = {}
        self.health_check_interval = 60  # seconds
        self.max_consecutive_failures = 3
        self.response_time_threshold = 30.0  # seconds
        self.error_rate_threshold = 0.2  # 20%
        
        # Performance tracking
        self.response_times = {}
        self.error_counts = {}
        self.request_counts = {}
        
        # Initialize health tracking for all models
        self._initialize_health_tracking()

    def _get_backup_models(self) -> List[str]:
        """Get list of backup models."""
        # Define backup models in order of preference
        backup_models = [
            "meta-llama-4-maverick-17b",  # Primary backup
            "llama-3-70b-8192",           # Secondary backup
            "mixtral-8x7b-32768",         # Tertiary backup
        ]
        
        # Remove primary model from backup list if present
        return [model for model in backup_models if model != self.primary_model]

    def _initialize_health_tracking(self):
        """Initialize health tracking for all models."""
        all_models = [self.primary_model] + self.backup_models
        
        for model in all_models:
            self.model_health[model] = ModelHealth(
                model_name=model,
                status=ModelStatus.HEALTHY,
                response_time=0.0,
                error_rate=0.0,
                last_check=datetime.now(),
                consecutive_failures=0,
                total_requests=0,
                successful_requests=0
            )
            
            self.response_times[model] = []
            self.error_counts[model] = 0
            self.request_counts[model] = 0

    async def execute_with_fallback(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with automatic failover to backup models.
        
        Args:
            operation: Function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            LLMInterfaceError: If all models fail
        """
        models_to_try = self._get_models_in_priority_order()
        
        for model in models_to_try:
            try:
                self.logger.info(f"Attempting operation with model: {model}")
                
                # Update current model
                self.current_model = model
                
                # Execute operation
                start_time = time.time()
                result = await operation(*args, **kwargs)
                response_time = time.time() - start_time
                
                # Record success
                self._record_success(model, response_time)
                
                self.logger.info(f"Operation successful with model: {model}")
                return result
                
            except Exception as e:
                # Record failure
                self._record_failure(model, str(e))
                
                self.logger.warning(f"Operation failed with model {model}: {e}")
                
                # Check if we should continue to next model
                if self._should_continue_to_next_model(model):
                    continue
                else:
                    break
        
        # All models failed
        raise LLMInterfaceError("All available models failed")

    def _get_models_in_priority_order(self) -> List[str]:
        """Get models in priority order based on health status."""
        # Start with current model if it's healthy
        if self._is_model_healthy(self.current_model):
            models = [self.current_model]
        else:
            models = []
        
        # Add other healthy models
        for model in [self.primary_model] + self.backup_models:
            if model != self.current_model and self._is_model_healthy(model):
                models.append(model)
        
        # If no healthy models, try all models in priority order
        if not models:
            models = [self.primary_model] + self.backup_models
        
        return models

    def _is_model_healthy(self, model: str) -> bool:
        """Check if a model is healthy."""
        if model not in self.model_health:
            return False
        
        health = self.model_health[model]
        
        # Check if model is offline
        if health.status == ModelStatus.OFFLINE:
            return False
        
        # Check consecutive failures
        if health.consecutive_failures >= self.max_consecutive_failures:
            return False
        
        # Check error rate
        if health.error_rate > self.error_rate_threshold:
            return False
        
        # Check response time
        if health.response_time > self.response_time_threshold:
            return False
        
        return True

    def _record_success(self, model: str, response_time: float):
        """Record successful operation for a model."""
        if model not in self.model_health:
            return
        
        health = self.model_health[model]
        
        # Update counters
        health.total_requests += 1
        health.successful_requests += 1
        health.consecutive_failures = 0
        health.last_check = datetime.now()
        
        # Update response time tracking
        self.response_times[model].append(response_time)
        if len(self.response_times[model]) > 100:  # Keep last 100 measurements
            self.response_times[model].pop(0)
        
        # Calculate average response time
        if self.response_times[model]:
            health.response_time = statistics.mean(self.response_times[model])
        
        # Calculate error rate
        health.error_rate = 1 - (health.successful_requests / health.total_requests)
        
        # Update status
        health.status = self._calculate_status(health)
        
        # Update request count
        self.request_counts[model] = health.total_requests

    def _record_failure(self, model: str, error_message: str):
        """Record failed operation for a model."""
        if model not in self.model_health:
            return
        
        health = self.model_health[model]
        
        # Update counters
        health.total_requests += 1
        health.consecutive_failures += 1
        health.last_check = datetime.now()
        
        # Update error count
        self.error_counts[model] += 1
        
        # Calculate error rate
        health.error_rate = 1 - (health.successful_requests / health.total_requests)
        
        # Update status
        health.status = self._calculate_status(health)
        
        # Log the failure
        self.logger.error(f"Model {model} failure: {error_message}")

    def _calculate_status(self, health: ModelHealth) -> ModelStatus:
        """Calculate model status based on health metrics."""
        if health.consecutive_failures >= self.max_consecutive_failures:
            return ModelStatus.OFFLINE
        elif health.error_rate > self.error_rate_threshold:
            return ModelStatus.UNHEALTHY
        elif health.response_time > self.response_time_threshold:
            return ModelStatus.DEGRADED
        else:
            return ModelStatus.HEALTHY

    def _should_continue_to_next_model(self, failed_model: str) -> bool:
        """Determine if we should continue to the next model after a failure."""
        # Always continue if there are other models available
        available_models = [self.primary_model] + self.backup_models
        remaining_models = [m for m in available_models if m != failed_model]
        
        return len(remaining_models) > 0

    async def health_check(self, model: str) -> bool:
        """
        Perform health check on a specific model.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is healthy, False otherwise
        """
        try:
            # Simple health check - could be enhanced with actual API call
            start_time = time.time()
            
            # Simulate health check (replace with actual API call)
            await asyncio.sleep(0.1)
            
            response_time = time.time() - start_time
            
            # Record health check result
            self._record_success(model, response_time)
            
            return True
            
        except Exception as e:
            self._record_failure(model, f"Health check failed: {e}")
            return False

    async def perform_health_checks(self):
        """Perform health checks on all models."""
        self.logger.info("Performing health checks on all models")
        
        for model in [self.primary_model] + self.backup_models:
            try:
                is_healthy = await self.health_check(model)
                status = "healthy" if is_healthy else "unhealthy"
                self.logger.info(f"Model {model} health check: {status}")
            except Exception as e:
                self.logger.error(f"Health check failed for {model}: {e}")

    def get_model_status(self, model: str) -> Optional[ModelHealth]:
        """Get health status of a specific model."""
        return self.model_health.get(model)

    def get_all_model_status(self) -> Dict[str, ModelHealth]:
        """Get health status of all models."""
        return self.model_health.copy()

    def get_best_model(self) -> str:
        """Get the best performing model based on health metrics."""
        best_model = self.primary_model
        best_score = 0.0
        
        for model, health in self.model_health.items():
            if health.status == ModelStatus.OFFLINE:
                continue
            
            # Calculate score based on health metrics
            score = self._calculate_model_score(health)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        return best_model

    def _calculate_model_score(self, health: ModelHealth) -> float:
        """Calculate performance score for a model."""
        if health.status == ModelStatus.OFFLINE:
            return 0.0
        
        # Base score from status
        status_scores = {
            ModelStatus.HEALTHY: 1.0,
            ModelStatus.DEGRADED: 0.7,
            ModelStatus.UNHEALTHY: 0.3,
            ModelStatus.OFFLINE: 0.0
        }
        
        base_score = status_scores[health.status]
        
        # Adjust for response time (lower is better)
        response_time_factor = max(0.1, 1.0 - (health.response_time / self.response_time_threshold))
        
        # Adjust for error rate (lower is better)
        error_rate_factor = 1.0 - health.error_rate
        
        # Adjust for consecutive failures
        failure_factor = max(0.1, 1.0 - (health.consecutive_failures / self.max_consecutive_failures))
        
        # Calculate final score
        final_score = base_score * response_time_factor * error_rate_factor * failure_factor
        
        return final_score

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        metrics = {
            "current_model": self.current_model,
            "primary_model": self.primary_model,
            "backup_models": self.backup_models,
            "model_health": {},
            "overall_health": "healthy"
        }
        
        # Add health metrics for each model
        for model, health in self.model_health.items():
            metrics["model_health"][model] = {
                "status": health.status.value,
                "response_time": health.response_time,
                "error_rate": health.error_rate,
                "consecutive_failures": health.consecutive_failures,
                "total_requests": health.total_requests,
                "successful_requests": health.successful_requests,
                "score": self._calculate_model_score(health)
            }
        
        # Determine overall health
        healthy_models = sum(1 for h in self.model_health.values() if h.status != ModelStatus.OFFLINE)
        total_models = len(self.model_health)
        
        if healthy_models == 0:
            metrics["overall_health"] = "offline"
        elif healthy_models < total_models:
            metrics["overall_health"] = "degraded"
        else:
            metrics["overall_health"] = "healthy"
        
        return metrics

    def reset_model_health(self, model: str):
        """Reset health metrics for a specific model."""
        if model in self.model_health:
            self.model_health[model] = ModelHealth(
                model_name=model,
                status=ModelStatus.HEALTHY,
                response_time=0.0,
                error_rate=0.0,
                last_check=datetime.now(),
                consecutive_failures=0,
                total_requests=0,
                successful_requests=0
            )
            
            self.response_times[model] = []
            self.error_counts[model] = 0
            self.request_counts[model] = 0
            
            self.logger.info(f"Reset health metrics for model: {model}")

    def set_model_offline(self, model: str):
        """Manually set a model as offline."""
        if model in self.model_health:
            self.model_health[model].status = ModelStatus.OFFLINE
            self.logger.warning(f"Manually set model {model} as offline")

    def set_model_online(self, model: str):
        """Manually set a model as online."""
        if model in self.model_health:
            self.model_health[model].status = ModelStatus.HEALTHY
            self.model_health[model].consecutive_failures = 0
            self.logger.info(f"Manually set model {model} as online") 