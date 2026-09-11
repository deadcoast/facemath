from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    AnalysisBundle,
    CanonicalPair,
    Direction,
    FacePair,
    MetricResult,
    Provenance,
)
from .deformation import (
    optical_flow_farneback,
    smooth_deformation_energy,
    strain_energy,
)
from .geometry import (
    apply_homography,
    distance_matrix_error,
    epsilon_isometry_distortion,
    generalized_procrustes,
    relational_graph,
    similarity_transform,
    spectral_graph_distance,
)
from .signal import (
    difference_field,
    fourier_phase_distance,
    laplacian_pyramid_distance,
    residual_entropy,
    scattering_distance,
    structure_tensor_distance,
)
from .topology import gromov_relational_distortion, persistence_distance_0d


@dataclass
class GeometryPipeline:
    """One system: every analyzer consumes the same canonical pair and writes one ledger."""

    def canonicalize(self, bundle: AnalysisBundle) -> CanonicalPair:
        p = bundle.pair
        if p.landmarks_a is None or p.landmarks_b is None:
            c = CanonicalPair(p.a, p.b, p.landmarks_a, p.landmarks_b)
            bundle.canonical = c
            return c
        m, _, rmse = similarity_transform(p.landmarks_a.xy, p.landmarks_b.xy)
        # Warp A into B's raster coordinates.
        import cv2

        warped = cv2.warpAffine(
            p.a.rgb, m[:2].astype(np.float32), (p.b.rgb.shape[1], p.b.rgb.shape[0])
        )
        from .core import Frame, LandmarkSet

        la = LandmarkSet(
            apply_homography(p.landmarks_a.xy, m),
            p.landmarks_a.names,
            p.landmarks_a.confidence,
        )
        c = CanonicalPair(
            Frame(warped, p.a.timestamp, p.a.source), p.b, la, p.landmarks_b, m, rmse
        )
        bundle.canonical = c
        return c

    def run(self, pair: FacePair) -> AnalysisBundle:
        b = AnalysisBundle(pair=pair)
        c = self.canonicalize(b)
        a, bf = c.a.gray, c.b.gray
        prov = lambda op, params={}: Provenance(op, (c.a.digest, c.b.digest), params)

        # Signal-domain evidence: all registered to same coordinate system.
        b.ledger.add(
            MetricResult(
                "residual_rms",
                float(np.sqrt(np.mean((a - bf) ** 2))),
                "normalized_intensity",
                Direction.LOWER_IS_MORE_SIMILAR,
                "signal",
                "RMS registered residual",
                prov("difference_field"),
            )
        )
        b.ledger.add(
            MetricResult(
                "residual_entropy",
                residual_entropy(difference_field(a, bf)),
                "bits",
                Direction.LOWER_IS_MORE_SIMILAR,
                "signal",
                "Entropy of registered residual",
                prov("residual_entropy"),
            )
        )
        b.ledger.add(
            MetricResult(
                "structure_tensor_distance",
                structure_tensor_distance(a, bf),
                "tensor_rmse",
                Direction.LOWER_IS_MORE_SIMILAR,
                "signal",
                "Local edge/orientation structure disagreement",
                prov("structure_tensor"),
            )
        )
        b.ledger.add(
            MetricResult(
                "phase_distance",
                fourier_phase_distance(a, bf),
                "radians_rms",
                Direction.LOWER_IS_MORE_SIMILAR,
                "signal",
                "Fourier phase disagreement",
                prov("fourier_phase"),
            )
        )
        pyr = laplacian_pyramid_distance(a, bf)
        b.ledger.add(
            MetricResult(
                "multiscale_laplacian",
                pyr.tolist(),
                "rms_per_scale",
                Direction.DESCRIPTIVE,
                "signal",
                "Scale-specific appearance/structure residual",
                prov("laplacian_pyramid"),
            )
        )
        b.ledger.add(
            MetricResult(
                "scattering_distance",
                scattering_distance(a, bf),
                "feature_rmse",
                Direction.LOWER_IS_MORE_SIMILAR,
                "signal",
                "Deformation-stable Gabor scattering-like distance",
                prov("gabor_scattering"),
            )
        )

        # Dense deformation is derived from the same canonical images.
        flow = optical_flow_farneback(a, bf)
        b.artifacts["dense_flow"] = flow.uv
        b.ledger.add(
            MetricResult(
                "flow_mean_magnitude",
                float(flow.magnitude.mean()),
                "pixels",
                Direction.LOWER_IS_MORE_SIMILAR,
                "deformation",
                "Dense apparent displacement after canonical registration",
                prov("farneback"),
            )
        )
        b.ledger.add(
            MetricResult(
                "strain_energy",
                strain_energy(flow),
                "squared_gradient",
                Direction.LOWER_IS_MORE_SIMILAR,
                "deformation",
                "Translation-insensitive local deformation strain",
                prov("strain_tensor"),
            )
        )
        b.ledger.add(
            MetricResult(
                "smooth_deformation_energy",
                smooth_deformation_energy(flow),
                "energy",
                Direction.LOWER_IS_MORE_SIMILAR,
                "deformation",
                "Magnitude + first/second derivative deformation energy",
                prov("deformation_energy"),
            )
        )

        if c.landmarks_a is not None and c.landmarks_b is not None:
            x, y = c.landmarks_a.xy, c.landmarks_b.xy
            proc = generalized_procrustes(x, y)
            b.ledger.add(
                MetricResult(
                    "procrustes_distance",
                    proc.distance,
                    "normalized_frobenius",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "geometry",
                    "Translation/scale/rotation-normalized landmark distance",
                    prov("procrustes"),
                )
            )
            b.ledger.add(
                MetricResult(
                    "registration_rmse",
                    float(c.registration_rmse or 0),
                    "pixels",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "geometry",
                    "Similarity-transform landmark registration error",
                    prov("similarity_transform"),
                )
            )
            b.ledger.add(
                MetricResult(
                    "distance_matrix_error",
                    distance_matrix_error(proc.aligned_a, proc.aligned_b),
                    "normalized_distance_rmse",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "geometry",
                    "Relational landmark metric distortion",
                    prov("distance_matrix"),
                )
            )
            eps, distortion = epsilon_isometry_distortion(
                proc.aligned_a, proc.aligned_b
            )
            b.artifacts["relational_distortion_matrix"] = distortion
            b.ledger.add(
                MetricResult(
                    "epsilon_isometry",
                    eps,
                    "normalized_distance",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "geometry",
                    "Maximum pairwise metric distortion under correspondence",
                    prov("epsilon_isometry"),
                )
            )
            ga, gb = relational_graph(proc.aligned_a), relational_graph(proc.aligned_b)
            b.ledger.add(
                MetricResult(
                    "graph_spectral_distance",
                    spectral_graph_distance(ga, gb),
                    "spectral_rmse",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "relational",
                    "Landmark-graph Laplacian spectral distance",
                    prov("graph_spectrum"),
                )
            )
            b.ledger.add(
                MetricResult(
                    "persistence_0d_distance",
                    persistence_distance_0d(proc.aligned_a, proc.aligned_b),
                    "persistence_rmse",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "topology",
                    "0D Vietoris-Rips persistence distance",
                    prov("persistence_0d"),
                )
            )
            b.ledger.add(
                MetricResult(
                    "gromov_relational_distortion",
                    gromov_relational_distortion(proc.aligned_a, proc.aligned_b),
                    "squared_metric_distortion",
                    Direction.LOWER_IS_MORE_SIMILAR,
                    "relational",
                    "GW-style relational distortion under fixed correspondence",
                    prov("gromov_relational"),
                )
            )
        return b
