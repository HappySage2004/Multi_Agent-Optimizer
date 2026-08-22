"""M3 -- Booking Probability Model.

Unified win/loss training frame:
  - WON  (label=1): bookings, price = contracted_price_per_slot_per_day
  - LOST (label=0): lost_leads where loss_reason is price-driven
                    (price_too_high, budget_mismatch), price = quoted_price_per_slot_per_day

Controls: screen_size, screen_type, position, city_id, industry_vertical.

NOTE: daypart is deliberately excluded. `lost_leads` has no daypart/time_block_id field at
all (verified against the source), so including it would either force dropping every
negative example, or require an 'unknown' category on every negative example -- which
would let the model learn 'unknown daypart -> always lost' as a spurious artifact.
Daypart-level price variation is already captured upstream by the Price Band Engine (M2);
this model estimates the probability curve *given* a price already anchored to the right
daypart/segment band.

The critical validation this module must pass: the price coefficient must be NEGATIVE
after controlling for screen quality/city/industry. A positive coefficient means the
confounders aren't sufficiently controlled (premium screens both cost more AND book more)
and the model should not be trusted for pricing decisions.

Port note: the model, features, split and calibration are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from app.logging_utils import debug, info

PRICE_DRIVEN_LOSS_REASONS = ["price_too_high", "budget_mismatch"]


@dataclass
class TrainingReport:
    n_won: int
    n_lost: int
    price_coefficient: float
    price_coefficient_sign_ok: bool
    auc: float
    coefficients: dict
    mean_predicted_prob: float
    true_base_rate: float
    calibration_ok: bool


class BookingProbabilityModel:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.feature_cols_categorical = [
            "screen_size",
            "screen_type",
            "position",
            "city_id",
            "industry_vertical",
        ]
        self.training_report = None

    def _build_training_frame(self, bookings_df, lost_leads_df, screens_df):
        screens = screens_df.copy()
        screens["position"] = screens["position"].fillna("not_applicable")
        screens_lookup = screens.set_index("screen_id")[["screen_size", "screen_type", "position"]]

        # --- positive class: bookings ---
        won = bookings_df.merge(screens_lookup, on="screen_id", how="left")
        won = won[
            [
                "screen_size",
                "screen_type",
                "position",
                "city_id",
                "industry_vertical",
                "contracted_price_per_slot_per_day",
            ]
        ].rename(columns={"contracted_price_per_slot_per_day": "price"})
        won["label"] = 1

        # --- negative class: price-driven lost leads ---
        lost = lost_leads_df[lost_leads_df["loss_reason"].isin(PRICE_DRIVEN_LOSS_REASONS)].copy()
        lost = lost[lost["quoted_price_per_slot_per_day"].notnull()]
        lost = lost.merge(screens_lookup, left_on="anchor_screen_id", right_index=True, how="left")
        lost = lost[
            [
                "screen_size",
                "screen_type",
                "position",
                "city_id",
                "industry_vertical",
                "quoted_price_per_slot_per_day",
            ]
        ].rename(columns={"quoted_price_per_slot_per_day": "price"})
        lost["label"] = 0
        lost = lost.dropna(subset=["screen_size", "screen_type", "position"])

        frame = pd.concat([won, lost], ignore_index=True)
        frame["log_price"] = np.log(frame["price"])
        debug(
            f"booking probability: training frame {len(frame):,} rows — {len(won):,} won "
            f"vs {len(lost):,} price-driven lost ({len(won) / max(len(lost), 1):.0f}:1 "
            f"imbalance), price {frame['price'].min():,.2f}..{frame['price'].max():,.2f}"
        )
        return frame, len(won), len(lost)

    def fit(self, bookings_df, lost_leads_df, screens_df, random_state=42):
        """Two-stage fit, addressing the ~490:1 class imbalance properly.

        Stage 1 (coefficient direction): fit LogisticRegression with
        class_weight='balanced' on a train split. Balancing is necessary here -- an
        unweighted fit on this imbalance produces a WRONG-SIGNED price coefficient
        (verified empirically: +0.35 unweighted vs -1.18 to -1.43 balanced), because 393
        negative examples can't outweigh noise in the majority class under plain MLE.

        Stage 2 (probability calibration): balanced class weighting distorts absolute
        probabilities toward an artificial 50/50 prior. Re-calibrate on a separate
        held-out split using CalibratedClassifierCV (sigmoid / Platt scaling) against the
        TRUE (unweighted) label distribution. This preserves the correctly-signed ranking
        from Stage 1 while restoring realistic absolute probabilities.
        """
        frame, n_won, n_lost = self._build_training_frame(bookings_df, lost_leads_df, screens_df)

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = self.encoder.fit_transform(frame[self.feature_cols_categorical])
        X = np.hstack([frame[["log_price"]].values, X_cat])
        y = frame["label"].values

        # three-way split: train (fit balanced base) / calib (fit calibration) / test
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=random_state, stratify=y
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=random_state, stratify=y_rest
        )

        debug(
            f"booking probability: split train={len(y_train):,} calib={len(y_calib):,} "
            f"test={len(y_test):,}, {X.shape[1]} feature(s) after one-hot encoding"
        )

        base = LogisticRegression(max_iter=2000, class_weight="balanced")
        base.fit(X_train, y_train)

        price_coef = base.coef_[0][0]  # log_price is first column

        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        calibrated.fit(X_calib, y_calib)
        self.model = calibrated
        self._base_model = base  # kept for coefficient inspection

        test_probs = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, test_probs)
        mean_pred = float(test_probs.mean())
        true_rate = float(y_test.mean())  # rate of label=1 (won) in test split

        feature_names = ["log_price"] + list(
            self.encoder.get_feature_names_out(self.feature_cols_categorical)
        )
        coefficients = dict(zip(feature_names, base.coef_[0], strict=True))

        self.training_report = TrainingReport(
            n_won=n_won,
            n_lost=n_lost,
            price_coefficient=float(price_coef),
            price_coefficient_sign_ok=bool(price_coef < 0),
            auc=float(auc),
            coefficients=coefficients,
            mean_predicted_prob=mean_pred,
            true_base_rate=true_rate,
            calibration_ok=bool(abs(mean_pred - true_rate) < 0.01),
        )

        debug(
            f"booking probability: fitted price_coef={price_coef:+.4f} auc={auc:.4f}, "
            f"mean predicted {mean_pred:.4f} vs true base rate {true_rate:.4f} "
            f"(calibration_ok={self.training_report.calibration_ok})"
        )
        # The one check that must hold. A positive price coefficient means the confounders
        # are not controlled and the probability output is not fit to price off.
        if not self.training_report.price_coefficient_sign_ok:
            info(
                f"booking probability WARNING: price coefficient {price_coef:+.4f} is "
                f"positive (expected negative) — probabilities are not trustworthy"
            )
        if not self.training_report.calibration_ok:
            info(
                f"booking probability: calibration off by "
                f"{abs(mean_pred - true_rate):.4f} (mean predicted {mean_pred:.4f} vs "
                f"base rate {true_rate:.4f}) — absolute probabilities are indicative only"
            )
        return self.training_report

    def predict_proba(
        self, price, screen_size, screen_type, position, city_id, industry_vertical
    ) -> float:
        """Single-point prediction: P(booked | price, context)."""
        row = pd.DataFrame(
            [
                {
                    "screen_size": screen_size,
                    "screen_type": screen_type,
                    "position": position,
                    "city_id": city_id,
                    "industry_vertical": industry_vertical,
                }
            ]
        )
        X_cat = self.encoder.transform(row[self.feature_cols_categorical])
        X = np.hstack([[[np.log(price)]], X_cat])
        return float(self.model.predict_proba(X)[0, 1])

    def predict_proba_curve(
        self, price_range, screen_size, screen_type, position, city_id, industry_vertical
    ):
        """Vectorized version across a price range, for the argmax diagnostic in M4."""
        n = len(price_range)
        row = pd.DataFrame(
            [
                {
                    "screen_size": screen_size,
                    "screen_type": screen_type,
                    "position": position,
                    "city_id": city_id,
                    "industry_vertical": industry_vertical,
                }
            ]
            * n
        )
        X_cat = self.encoder.transform(row[self.feature_cols_categorical])
        log_prices = np.log(np.asarray(price_range)).reshape(-1, 1)
        X = np.hstack([log_prices, X_cat])
        return self.model.predict_proba(X)[:, 1]
