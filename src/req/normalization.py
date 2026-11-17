import numpy as np
import pandas as pd

from req.features import FEATURE_COLUMNS

class FeatureNormalizer:

    def __init__(self, clip_value=5.0):

        self.clip_value = clip_value

        self.medians = {}
        self.iqrs = {}

        self.fitted = False

        self.robust_features = [
            "Return1",
            "Return5",
            "Return20",
            "RPD10",
            "Volatility20",
            "RelativeVolume"
        ]

    # -----------------------------------------------
    # Apply transformations before scaling
    # -----------------------------------------------

    def _transform_raw(self, df):

        x = df[FEATURE_COLUMNS].copy().astype(float)

        # Positive, right-skewed features.
        x["Volatility20"] = np.log(
            x["Volatility20"].clip(lower=1e-8)
        )

        x["RelativeVolume"] = np.log1p(
            x["RelativeVolume"].clip(lower=0)
        )

        # Map RSI from [0, 100] to [-1, 1].
        x["RSI14"] = (x["RSI14"] - 50.0) / 50.0

        return x

    # -----------------------------------------------
    # Fit using training data only
    # -----------------------------------------------

    def fit(self, training_datasets):

        """
        training_datasets:
            A list of feature DataFrames, one per stock.

        IMPORTANT:
            Pass only training-period rows.
        """

        if isinstance(training_datasets, pd.DataFrame):
            training_datasets = [training_datasets]

        transformed = []

        for df in training_datasets:

            x = self._transform_raw(df)

            transformed.append(x[self.robust_features])

        # Pool observations from all training stocks.
        pooled = pd.concat(
            transformed,
            axis=0,
            ignore_index=True
        )

        for feature in self.robust_features:

            values = pooled[feature].to_numpy(dtype=float)

            values = values[np.isfinite(values)]

            if len(values) == 0:
                raise ValueError(
                    f"No valid training values for {feature}"
                )

            median = np.median(values)

            q25 = np.percentile(values, 25)
            q75 = np.percentile(values, 75)

            iqr = q75 - q25

            # Avoid division by zero.
            if iqr < 1e-8:
                iqr = 1.0

            self.medians[feature] = median
            self.iqrs[feature] = iqr

        self.fitted = True

        return self

    # -----------------------------------------------
    # Transform data using fitted statistics
    # -----------------------------------------------

    def transform(self, df):

        if not self.fitted:
            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        x = self._transform_raw(df)

        for feature in self.robust_features:

            x[feature] = (
                x[feature] - self.medians[feature]
            ) / self.iqrs[feature]

        # Clip extreme scaled values.
        x = x.clip(
            lower=-self.clip_value,
            upper=self.clip_value
        )

        x = x.replace([np.inf, -np.inf], np.nan)

        if x.isna().any().any():
            raise ValueError(
                "NaN or infinite values encountered during normalization."
            )

        # return x[FEATURE_COLUMNS]

        result = x[FEATURE_COLUMNS].copy()
        result.insert(0, "Date", df["Date"].values)
        return result

    # -----------------------------------------------
    # Convenience function
    # -----------------------------------------------

    def fit_transform(self, training_datasets):

        self.fit(training_datasets)

        if isinstance(training_datasets, pd.DataFrame):
            return self.transform(training_datasets)

        return [
            self.transform(df)
            for df in training_datasets
        ]