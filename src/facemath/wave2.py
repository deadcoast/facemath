from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    AnalysisBundle,
    CanonicalPair,
    Direction,
    FacePair,
    LandmarkSet,
    MetricResult,
    Provenance,
)
from .correspondence import (
    dense_correspondence_farneback,
    region_correspondence_summary,
    weighted_flow_statistics,
)
from .deformation import FlowField, smooth_deformation_energy, strain_energy
from .geometry import (
    distance_matrix_error,
    epsilon_isometry_distortion,
    generalized_procrustes,
    relational_graph,
    spectral_graph_distance,
)
from .quality import assess_state
from .regions import DEFAULT_SCHEMA, RegionSchema, build_regions, ellipse_face_mask
from .registration import (
    diagnose_registration,
    ransac_similarity_transform,
    warp_frame,
    weighted_similarity_transform,
)
from .signal import (
    difference_field,
    fourier_phase_distance,
    laplacian_pyramid_distance,
    residual_entropy,
    scattering_distance,
    structure_tensor_distance,
)
from .state import FacialState, RegistrationDiagnostics, SupportLevel
from .topology import gromov_relational_distortion, persistence_distance_0d


@dataclass(frozen=True)
class EngineConfig:
    ransac_threshold_px: float = 4.0
    ransac_trials: int = 512
    correspondence_fb_threshold_px: float = 1.75
    minimum_valid_support: float = 0.20
    region_schema: RegionSchema = DEFAULT_SCHEMA
    use_dense_correspondence: bool = True


class FacialGeometryEngine:
    """Wave-2 integrated engine.

    Canonical registration, anatomy, dense correspondence, quality gating, and
    evidence extraction share one immutable FacialState. An analyzer may not
    silently re-register or choose its own coordinate system.
    """

    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()

    @staticmethod
    def _prov(state: FacialState, op: str, params: dict | None = None) -> Provenance:
        return Provenance(op, (state.a.digest, state.b.digest), params or {})

    def build_state(
        self, pair: FacePair, face_mask: np.ndarray | None = None
    ) -> FacialState:
        a, b = pair.a, pair.b
        la, lb = pair.landmarks_a, pair.landmarks_b
        h, w = b.gray.shape
        if la is not None and lb is not None:
            conf = None
            if la.confidence is not None or lb.confidence is not None:
                ca = (
                    np.ones(len(la.xy))
                    if la.confidence is None
                    else np.asarray(la.confidence, float)
                )
                cb = (
                    np.ones(len(lb.xy))
                    if lb.confidence is None
                    else np.asarray(lb.confidence, float)
                )
                conf = np.minimum(ca, cb)
            m, inliers, _ = ransac_similarity_transform(
                la.xy, lb.xy, self.config.ransac_threshold_px, self.config.ransac_trials
            )
            # weighted refinement over RANSAC inliers; low-confidence points cannot dominate
            weights = (np.ones(len(la.xy)) if conf is None else conf) * inliers.astype(
                float
            )
            if weights.sum() > 1e-12:
                m, _, _ = weighted_similarity_transform(la.xy, lb.xy, weights)
            aw = warp_frame(a, m, (h, w))
            law = LandmarkSet(
                xy=(np.c_[la.xy, np.ones(len(la.xy))] @ m.T)[:, :2],
                names=la.names,
                confidence=la.confidence,
            )
            reg = diagnose_registration(la, lb, m, (h, w), inliers=inliers)
            if face_mask is None:
                face_mask = ellipse_face_mask((h, w), lb)
            regions = build_regions((h, w), lb, self.config.region_schema)
        else:
            m = np.eye(3)
            aw = a
            warnings = ("no_landmarks_registration_unverified",)
            reg = RegistrationDiagnostics(
                "identity_unverified",
                m,
                None,
                None,
                None,
                1.0,
                None,
                SupportLevel.DEGRADED,
                warnings,
            )
            law = la
            regions = {}
            if face_mask is None:
                face_mask = np.ones((h, w), bool)
        mask = np.asarray(face_mask, bool)
        corr = None
        if self.config.use_dense_correspondence:
            corr = dense_correspondence_farneback(
                aw.gray, b.gray, mask, self.config.correspondence_fb_threshold_px
            )
        return FacialState(
            aw, b, law, lb, reg, mask, regions, corr, metadata={"wave": 2}
        )

    @staticmethod
    def _masked_rms(
        a: np.ndarray, b: np.ndarray, m: np.ndarray, w: np.ndarray | None = None
    ) -> float:
        if not np.any(m):
            return float("nan")
        d = (a - b)[m]
        if w is None:
            return float(np.sqrt(np.mean(d * d)))
        ww = np.asarray(w, float)[m]
        return float(np.sqrt(np.sum(ww * d * d) / max(np.sum(ww), 1e-12)))

    def analyze_state(self, state: FacialState) -> AnalysisBundle:
        pair = FacePair(state.a, state.b, state.landmarks_a, state.landmarks_b)
        canonical = CanonicalPair(
            state.a,
            state.b,
            state.landmarks_a,
            state.landmarks_b,
            state.registration.matrix_a_to_b,
            state.registration.landmark_rmse_px,
        )
        bundle = AnalysisBundle(pair=pair, canonical=canonical)
        q = assess_state(state)
        bundle.artifacts["facial_state"] = state
        bundle.artifacts["quality_report"] = q
        bundle.artifacts["regions"] = {k: v.mask for k, v in state.regions.items()}
        if state.correspondence is not None:
            bundle.artifacts["correspondence"] = state.correspondence

        p = lambda op, params=None: self._prov(state, op, params)
        bundle.ledger.add(
            MetricResult(
                "quality_face_coverage",
                q.face_coverage,
                "fraction",
                Direction.DESCRIPTIVE,
                "quality",
                "Fraction of canonical raster admitted by the facial support mask",
                p("quality_gate"),
                tags=(q.support.value,),
            )
        )
        bundle.ledger.add(
            MetricResult(
                "quality_registration_overlap",
                q.registration_overlap,
                "fraction",
                Direction.HIGHER_IS_MORE_SIMILAR,
                "quality",
                "Raster overlap retained after canonical registration",
                p("quality_gate"),
                tags=(q.support.value,),
            )
        )
        if q.correspondence_support is not None:
            bundle.ledger.add(
                MetricResult(
                    "quality_correspondence_support",
                    q.correspondence_support,
                    "fraction",
                    Direction.HIGHER_IS_MORE_SIMILAR,
                    "quality",
                    "Fraction of facial support with cycle-consistent dense correspondence",
                    p("forward_backward_correspondence"),
                    tags=(q.support.value,),
                )
            )
            bundle.ledger.add(
                MetricResult(
                    "quality_correspondence_confidence",
                    q.correspondence_confidence or 0.0,
                    "mean_confidence",
                    Direction.HIGHER_IS_MORE_SIMILAR,
                    "quality",
                    "Mean confidence over valid dense facial correspondence",
                    p("forward_backward_correspondence"),
                    tags=(q.support.value,),
                )
            )

        # Insufficient support: preserve diagnostics but refuse substantive geometry claims.
        if q.support == SupportLevel.INSUFFICIENT:
            bundle.artifacts["analysis_refused"] = "insufficient geometric support"
            return bundle

        a, b = state.a.gray, state.b.gray
        m = state.valid_mask
        cw = (
            state.correspondence.confidence
            if state.correspondence is not None
            else np.ones_like(a)
        )
        bundle.ledger.add(
            MetricResult(
                "masked_residual_rms",
                self._masked_rms(a, b, m, cw),
                "normalized_intensity",
                Direction.LOWER_IS_MORE_SIMILAR,
                "signal",
                "Confidence-weighted registered residual restricted to valid face correspondence",
                p("masked_difference"),
            )
        )
        # Keep global structure metrics, but provenance records support mask usage limitation.
        bundle.ledger.add(
            MetricResult(
                "residual_entropy",
                residual_entropy(difference_field(a[m], b[m])),
                "bits",
                Direction.LOWER_IS_MORE_SIMILAR,
                "signal",
                "Residual entropy inside valid facial support",
                p("residual_entropy", {"masked": True}),
            )
        )
        bundle.ledger.add(
            MetricResult(
                "structure_tensor_distance",
                structure_tensor_distance(a, b),
                "tensor_rmse",
                Direction.LOWER_IS_MORE_SIMILAR,
                "signal",
                "Global local edge/orientation disagreement after canonical registration",
                p("structure_tensor"),
            )
        )
        bundle.ledger.add(
            MetricResult(
                "phase_distance",
                fourier_phase_distance(a, b),
                "radians_rms",
                Direction.LOWER_IS_MORE_SIMILAR,
                "signal",
                "Registered Fourier phase disagreement",
                p("fourier_phase"),
            )
        )
        bundle.ledger.add(
            MetricResult(
                "multiscale_laplacian",
                laplacian_pyramid_distance(a, b).tolist(),
                "rms_per_scale",
                Direction.DESCRIPTIVE,
                "signal",
                "Scale-resolved residual profile",
                p("laplacian_pyramid"),
            )
        )
        bundle.ledger.add(
            MetricResult(
                "scattering_distance",
                scattering_distance(a, b),
                "feature_rmse",
                Direction.LOWER_IS_MORE_SIMILAR,
                "signal",
                "Deformation-stable multiscale scattering-like distance",
                p("gabor_scattering"),
            )
        )

        if state.correspondence is not None:
            stats = weighted_flow_statistics(state.correspondence, state.face_mask)
            ff = FlowField(state.correspondence.uv)
            bundle.ledger.add(
                MetricResult(
                    "correspondence_mean_displacement",
                    stats["mean_magnitude"],
                    "pixels",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "deformation",
                    "Confidence-weighted dense displacement over facial support",
                    p("dense_correspondence"),
                )
            )
            bundle.ledger.add(
                MetricResult(
                    "correspondence_p90_displacement",
                    stats["p90_magnitude"],
                    "pixels",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "deformation",
                    "90th percentile dense displacement over valid facial support",
                    p("dense_correspondence"),
                )
            )
            bundle.ledger.add(
                MetricResult(
                    "strain_energy",
                    strain_energy(ff, mask=m),
                    "squared_gradient",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "deformation",
                    "Translation-insensitive local strain over valid facial support",
                    p("strain_tensor"),
                )
            )
            bundle.ledger.add(
                MetricResult(
                    "smooth_deformation_energy",
                    smooth_deformation_energy(ff, mask=m),
                    "energy",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "deformation",
                    "Regularized magnitude/gradient/curvature flow energy",
                    p("deformation_energy"),
                )
            )
            region_stats = region_correspondence_summary(
                state.correspondence, dict(state.regions)
            )
            bundle.artifacts["region_correspondence"] = region_stats
            for name, s in region_stats.items():
                if s["support_fraction"] <= 0:
                    continue
                bundle.ledger.add(
                    MetricResult(
                        f"region.{name}.mean_displacement",
                        s["mean_magnitude"],
                        "pixels",
                        Direction.LOWER_IS_MORE_SIMILAR,
                        "regional_geometry",
                        f"Dense displacement for anatomical region {name}",
                        p("regional_correspondence", {"region": name}),
                        tags=(name,),
                    )
                )
                bundle.ledger.add(
                    MetricResult(
                        f"region.{name}.confidence",
                        s["mean_confidence"],
                        "mean_confidence",
                        Direction.HIGHER_IS_MORE_SIMILAR,
                        "quality",
                        f"Correspondence confidence for anatomical region {name}",
                        p("regional_correspondence", {"region": name}),
                        tags=(name,),
                    )
                )

        if state.landmarks_a is not None and state.landmarks_b is not None:
            x, y = state.landmarks_a.xy, state.landmarks_b.xy
            proc = generalized_procrustes(x, y)
            bundle.ledger.add(
                MetricResult(
                    "procrustes_distance",
                    proc.distance,
                    "normalized_frobenius",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "geometry",
                    "Similarity-normalized landmark configuration distance",
                    p("procrustes"),
                )
            )
            if state.registration.landmark_rmse_px is not None:
                bundle.ledger.add(
                    MetricResult(
                        "registration_rmse",
                        state.registration.landmark_rmse_px,
                        "pixels",
                        Direction.LOWER_IS_MORE_SIMILAR,
                        "quality",
                        "Landmark error remaining after robust canonical registration",
                        p("registration"),
                    )
                )
            bundle.ledger.add(
                MetricResult(
                    "distance_matrix_error",
                    distance_matrix_error(proc.aligned_a, proc.aligned_b),
                    "normalized_distance_rmse",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "geometry",
                    "Pairwise relational landmark distortion",
                    p("distance_matrix"),
                )
            )
            eps, dist = epsilon_isometry_distortion(proc.aligned_a, proc.aligned_b)
            bundle.artifacts["relational_distortion_matrix"] = dist
            bundle.ledger.add(
                MetricResult(
                    "epsilon_isometry",
                    eps,
                    "normalized_distance",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "geometry",
                    "Worst normalized pairwise metric distortion",
                    p("epsilon_isometry"),
                )
            )
            ga, gb = relational_graph(proc.aligned_a), relational_graph(proc.aligned_b)
            bundle.ledger.add(
                MetricResult(
                    "graph_spectral_distance",
                    spectral_graph_distance(ga, gb),
                    "spectral_rmse",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "relational",
                    "Relational graph spectrum disagreement",
                    p("graph_spectrum"),
                )
            )
            bundle.ledger.add(
                MetricResult(
                    "persistence_0d_distance",
                    persistence_distance_0d(proc.aligned_a, proc.aligned_b),
                    "persistence_rmse",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "topology",
                    "0D persistence disagreement",
                    p("persistence_0d"),
                )
            )
            bundle.ledger.add(
                MetricResult(
                    "gromov_relational_distortion",
                    gromov_relational_distortion(proc.aligned_a, proc.aligned_b),
                    "squared_metric_distortion",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "relational",
                    "Gromov-style relational distortion",
                    p("gromov_relational"),
                )
            )
        return bundle

    def run(
        self, pair: FacePair, face_mask: np.ndarray | None = None
    ) -> AnalysisBundle:
        return self.analyze_state(self.build_state(pair, face_mask))
