"""Health check module for monitoring system health."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    check_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime


class HealthMonitor:
    """Monitor system health and data quality."""
    
    def __init__(self, output_dir: Path = Path("outputs")):
        self.output_dir = output_dir
        self.results: List[HealthCheckResult] = []
    
    def check_data_freshness(self, max_age_days: int = 7) -> HealthCheckResult:
        """Check if output data is fresh."""
        try:
            files = list(self.output_dir.glob("anomalies_*.csv"))
            if not files:
                return HealthCheckResult(
                    "data_freshness", "unhealthy", 
                    "No output files found", datetime.now()
                )
            
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
            
            if file_age.days > max_age_days:
                return HealthCheckResult(
                    "data_freshness", "unhealthy",
                    f"Data is {file_age.days} days old", datetime.now()
                )
            return HealthCheckResult(
                "data_freshness", "healthy",
                f"Data is {file_age.days} days old", datetime.now()
            )
        except Exception as e:
            return HealthCheckResult(
                "data_freshness", "unhealthy",
                f"Check failed: {str(e)}", datetime.now()
            )
    
    def run_all_checks(self) -> Dict[str, str]:
        """Run all health checks and return overall status."""
        self.results = [
            self.check_data_freshness(),
        ]
        
        if any(r.status == "unhealthy" for r in self.results):
            return {"status": "unhealthy", "results": self.results}
        elif any(r.status == "degraded" for r in self.results):
            return {"status": "degraded", "results": self.results}
        return {"status": "healthy", "results": self.results}


if __name__ == "__main__":
    monitor = HealthMonitor()
    results = monitor.run_all_checks()
    print(json.dumps(results, default=str, indent=2))

