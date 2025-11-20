"""Model scoring module for anomaly detection using IsolationForest.

This module provides the ModelScorer class which handles:
- Feature preparation and scaling with StandardScaler
- IsolationForest model training
- Anomaly score and label generation
- Model artifact persistence for reproducibility
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.exceptions import (
    ConfigurationError,
    ModelScoringError,
)

logger = logging.getLogger(__name__)


class ModelScorer:
    """Score customer features using IsolationForest for anomaly detection.

    The ModelScorer implements the complete scoring pipeline:
    1. Feature matrix preparation (extract numeric columns, handle missing values)
    2. Feature scaling with StandardScaler (zero mean, unit variance)
    3. IsolationForest training with configured hyperparameters
    4. Anomaly score and label generation
    5. Model artifact persistence for reproducibility

    Attributes:
        config: IsolationForest configuration from modelconfig.yaml
        logger: Logger instance for tracking scoring operations
        scaler: Fitted StandardScaler instance (None until fit_and_score called)
        model: Fitted IsolationForest instance (None until fit_and_score called)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with modelconfig.yaml hyperparameters.

        Args:
            config: Configuration dictionary containing 'isolationforest' key

        Raises:
            ConfigurationError: If required configuration is missing or invalid
        """
        if "isolationforest" not in config:
            raise ConfigurationError("Missing 'isolationforest' in configuration")

        self.config = config["isolationforest"]
        self.logger = logging.getLogger("models.scorer")
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[IsolationForest] = None

        # Validate required hyperparameters
        required_params = [
            "nestimators",
            "contamination",
            "maxsamples",
            "randomstate",
            "maxfeatures",
        ]
        missing_params = [p for p in required_params if p not in self.config]
        if missing_params:
            raise ConfigurationError(
                f"Missing required IsolationForest parameters: {missing_params}"
            )

        self.logger.info("ModelScorer initialized")

    def fit_and_score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Fit IsolationForest on features and score all customers.

        This is the main orchestration method that executes the complete pipeline:
        1. Prepare feature matrix (extract numeric columns)
        2. Fit scaler and transform features
        3. Fit IsolationForest model
        4. Score anomalies and generate labels
        5. Return DataFrame with added anomaly_score and anomaly_label columns

        Args:
            features_df: Feature matrix from FeatureBuilder with customer_id,
                        reporting_week, and numeric feature columns

        Returns:
            DataFrame with added columns:
            - anomaly_score (float): Decision function score (lower = more anomalous)
            - anomaly_label (int): Binary label (1 = normal, -1 = anomaly)

        Raises:
            ModelScoringError: If feature matrix is empty or scoring fails

        Side Effects:
            - Sets self.scaler: StandardScaler fitted on features
            - Sets self.model: IsolationForest fitted on scaled features
            - Sets self.feature_names: List of feature column names used in training
            - Sets self.row_count: Number of valid rows used for training
            - Sets self.anomaly_count: Number of anomalies detected
        """
        if features_df.empty:
            raise ModelScoringError("Cannot score empty feature matrix")

        row_count = len(features_df)
        self.logger.info(f"Starting scoring pipeline for {row_count} customers")

        # Prepare feature matrix
        X, feature_cols, valid_mask = self.prepare_feature_matrix(features_df)
        valid_count = valid_mask.sum()
        self.logger.info(
            f"Prepared feature matrix: {valid_count}/{row_count} valid rows, "
            f"{len(feature_cols)} features"
        )

        # Store training metrics as instance attributes for artifact persistence
        self.feature_names = feature_cols
        self.row_count = int(valid_count)

        # Fit scaler and transform features
        X_scaled = self.fit_scaler(X)
        self.logger.info("Fitted StandardScaler")

        # Fit IsolationForest
        self.fit_isolation_forest(X_scaled)

        # Score anomalies
        scores, labels = self.score_anomalies(X_scaled)
        anomaly_count = int((labels == -1).sum())
        anomaly_pct = (anomaly_count / valid_count) * 100

        # Store anomaly count for artifact persistence
        self.anomaly_count = anomaly_count

        self.logger.info(
            f"Scored {valid_count} valid customers, identified "
            f"{anomaly_count} anomalies ({anomaly_pct:.1f}%)"
        )

        # Add scores and labels to DataFrame
        # Use Option 2: Fill dropped rows with sentinel values
        # (np.nan for scores, 0 for labels)
        result_df = features_df.copy()
        result_df["anomaly_score"] = np.nan
        result_df["anomaly_label"] = (
            0  # 0 = unknown/invalid (neither normal=1 nor anomaly=-1)
        )
        result_df.loc[valid_mask, "anomaly_score"] = scores
        result_df.loc[valid_mask, "anomaly_label"] = labels

        return result_df

    def prepare_feature_matrix(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Extract numeric features and handle missing values.

        Selects all numeric columns except customer_id and reporting_week.
        Handles missing values based on configuration (currently drops rows).

        Args:
            df: DataFrame with features from FeatureBuilder

        Returns:
            Tuple of (feature_matrix, feature_column_names, valid_mask):
            - feature_matrix: Cleaned feature array (M rows after dropping NaN)
            - feature_column_names: List of feature column names
            - valid_mask: Boolean array (N elements) indicating valid rows

        Raises:
            ModelScoringError: If no numeric features found or all values are NaN
        """
        # Exclude identifier columns
        exclude_cols = ["customer_id", "reporting_week"]
        numeric_cols = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

        if not numeric_cols:
            raise ModelScoringError("No numeric features found in DataFrame")

        X = df[numeric_cols].values

        # Check for NaN values
        if np.isnan(X).all():
            raise ModelScoringError("All feature values are NaN")

        # Drop rows with any NaN values (conservative approach)
        nan_mask = np.isnan(X).any(axis=1)
        valid_mask = ~nan_mask  # Track which rows are valid

        if nan_mask.any():
            nan_count = nan_mask.sum()
            self.logger.warning(
                f"Dropping {nan_count} rows with missing values "
                f"({(nan_count/len(X))*100:.1f}%)"
            )
            X = X[valid_mask]

            if X.shape[0] == 0:
                raise ModelScoringError(
                    "No valid rows remaining after dropping missing values"
                )

        return X, numeric_cols, valid_mask

    def fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """Fit StandardScaler and transform features.

        Transforms features to zero mean and unit variance. This ensures all
        features contribute equally to the IsolationForest model regardless of
        their original scale.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Scaled feature matrix with mean ≈ 0, std ≈ 1

        Side Effects:
            Sets self.scaler with fitted StandardScaler
        """
        from typing import cast

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        return cast(np.ndarray, X_scaled)

    def fit_isolation_forest(self, X_scaled: np.ndarray) -> None:
        """Fit IsolationForest with configured hyperparameters.

        Initializes and trains the IsolationForest model on scaled features.
        The model learns to isolate anomalies by building an ensemble of
        isolation trees.

        Args:
            X_scaled: Scaled feature matrix from fit_scaler

        Side Effects:
            Sets self.model with fitted IsolationForest instance

        Raises:
            ModelScoringError: If model fitting fails
        """
        try:
            # Map config keys to sklearn parameter names
            n_estimators = self.config["nestimators"]
            max_samples = self.config["maxsamples"]
            contamination = self.config["contamination"]
            max_features = self.config["maxfeatures"]
            random_state = self.config["randomstate"]
            bootstrap = self.config.get("bootstrap", False)

            # Warn if contamination is unusually high
            if contamination is not None and contamination > 0.2:
                self.logger.warning(
                    f"High contamination value: {contamination} > 0.2 "
                    f"may affect anomaly detection"
                )

            self.model = IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=contamination,
                max_features=max_features,
                random_state=random_state,
                bootstrap=bootstrap,
            )

            self.model.fit(X_scaled)

            self.logger.info(
                f"IsolationForest trained: n_estimators={n_estimators}, "
                f"contamination={contamination}"
            )

        except Exception as e:
            raise ModelScoringError(
                f"IsolationForest training failed: {type(e).__name__}: {e}"
            ) from e

    def score_anomalies(self, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate anomaly scores and labels.

        Uses the fitted IsolationForest to compute:
        - Decision function scores: continuous values (lower = more anomalous)
        - Predict labels: binary values (-1 = anomaly, 1 = normal)

        Args:
            X_scaled: Scaled feature matrix

        Returns:
            Tuple of (scores, labels):
            - scores: Anomaly scores (float array, lower = more anomalous)
            - labels: Anomaly labels (int array, -1 = anomaly, 1 = normal)

        Raises:
            ModelScoringError: If model is not fitted or scoring fails
        """
        if self.model is None:
            raise ModelScoringError(
                "Model not fitted. Call fit_isolation_forest first."
            )

        try:
            scores = self.model.decision_function(X_scaled)
            labels = self.model.predict(X_scaled)
            return scores, labels

        except Exception as e:
            raise ModelScoringError(f"Anomaly scoring failed: {e}") from e

    def save_artifacts(self, output_dir: str, reporting_week: str) -> None:
        """Persist model artifacts for reproducibility and auditing.

        Creates a versioned directory structure and saves:
        - scaler.pkl: Fitted StandardScaler
        - model.pkl: Fitted IsolationForest
        - metadata.json: Feature names, config snapshot, training metrics

        Args:
            output_dir: Base output directory (e.g., './outputs')
            reporting_week: Reporting week for versioning (e.g., '2025-11-18')

        Raises:
            ModelScoringError: If model/scaler not fitted or save fails
        """
        if self.model is None or self.scaler is None:
            raise ModelScoringError(
                "Model and scaler must be fitted before saving artifacts"
            )

        # Create versioned directory
        artifact_dir = os.path.join(output_dir, "model_artifacts", reporting_week)
        os.makedirs(artifact_dir, exist_ok=True)

        try:
            # Save scaler
            scaler_path = os.path.join(artifact_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

            # Save model
            model_path = os.path.join(artifact_dir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            # Save metadata with training metrics
            metadata = {
                "training_date": datetime.now().isoformat(),
                "reporting_week": reporting_week,
                "config": self.config,
                "sklearn_version": self._get_sklearn_version(),
                "feature_names": self.feature_names,
                "row_count": self.row_count,
                "anomaly_count": self.anomaly_count,
            }

            metadata_path = os.path.join(artifact_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Saved model artifacts to {artifact_dir}")

        except Exception as e:
            raise ModelScoringError(f"Failed to save artifacts: {e}") from e

    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version for metadata."""
        try:
            from typing import cast

            import sklearn

            return cast(str, sklearn.__version__)
        except Exception:
            return "unknown"
