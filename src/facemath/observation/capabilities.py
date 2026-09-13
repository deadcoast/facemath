"""Honest inherited-mathematics inventory, not a string-presence coverage test.

Presence of a callable is necessary but not sufficient for mathematical
fidelity. Known non-equivalences are recorded explicitly and remain open work.
"""
from __future__ import annotations

from ..coverage import SECTION_TO_IMPLEMENTATION

KNOWN_NON_EQUIVALENCES = {
    34: ('partial', '0D merge deaths are not full persistent homology/bottleneck distance'),
    39: ('missing_exact', 'covariance whitening is not the affine-invariant SPD log metric'),
    45: ('conceptual_or_telemetry', 'cosine similarity is not an implementation of feature superposition'),
    48: ('partial', 'cosine is a primitive, not a recovered concept-direction probe'),
    50: ('partial', 'finite difference of a supplied scalar callable is not historical model Jacobian access'),
    51: ('missing_exact', 'gradient alone does not implement Jacobian singular-mode decomposition'),
    53: ('partial', 'Euclidean path energy is not a general Riemannian geodesic solver'),
    54: ('partial', 'path curvature does not construct or identify disentangled representation manifolds'),
    58: ('partial', 'Hungarian one-to-one assignment is not general mass-constrained optimal transport'),
    59: ('partial', 'position assignment omits the stated mixed feature/spatial transport problem'),
    60: ('partial', 'fixed-coupling relational distortion is not optimization of Gromov-Wasserstein transport'),
    66: ('missing_exact', 'percentile registry does not implement Stevens power-law magnitude estimation'),
    67: ('missing_exact', 'empirical ranks are not Bayesian posterior probabilities'),
    68: ('missing_exact', 'empirical ranks are not likelihood ratios'),
    69: ('missing_exact', 'empirical ranks are not dependent-evidence Bayes factor fusion'),
    77: ('missing_exact', 'CKA is not InfoNCE or contrastive representation training'),
    86: ('surrogate', 'Gabor summary is not a full spatial wavelet coefficient system'),
    87: ('surrogate', 'Gabor descriptor is not a Mallat scattering transform'),
    88: ('partial', 'deformation energy does not establish the cited deformation stability bound'),
    91: ('surrogate', 'TPS bending proxy is not a calibrated anatomical deformation field; Wave 3 provides a separate PWA field'),
    95: ('surrogate', 'compression-based mutual information is not computable exact Kolmogorov complexity'),
    96: ('conceptual', 'generic mutual-information estimation does not demonstrate geometric sufficiency'),
    97: ('missing_exact', 'ordinary pipeline is not an identified causal generative face model'),
    98: ('partial', 'occlusion is not controlled counterfactual 3D nuisance-factor rendering'),
    99: ('missing_exact', 'vertex normals do not implement a fitted 3D morphable model'),
    100: ('missing_exact', 'vertex normals do not implement camera projection/intrinsic calibration'),
    101: ('missing_exact', 'vertex normals do not implement depth recovery or epipolar inference'),
    103: ('missing_exact', 'Laplace-Beltrami eigensystem is not principal/mean/Gaussian curvature estimation'),
    104: ('partial', 'cotangent discretization is not a general surface geodesic solver'),
    105: ('partial', 'spectral distance is not an isometry proof; isospectral shapes exist'),
    123: ('missing_exact', 'GW-style objective does not implement higher-order relation tensors'),
    124: ('partial', '0D persistence is not a general simplicial-complex implementation'),
    126: ('conceptual', 'a relational graph does not implement general categorical morphism laws'),
    130: ('partial', 'inherited svcca name implements CCA without the stated SVD-reduction stage'),
    153: ('misindexed', 'RPCA mapping is not the ordinary temporal PCA described in section 153'),
    154: ('misindexed', 'DMD mapping is not RPCA described in section 154'),
    155: ('misindexed', 'CCA mapping is not DMD described in section 155'),
    157: ('missing_exact', 'mutual-information matrix is not partial least squares'),
    159: ('conceptual', 'mutual-information estimator is not information-bottleneck training'),
}


def inventory() -> dict:
    sections = []
    for number in range(181):
        implementation = SECTION_TO_IMPLEMENTATION.get(number)
        if number < 20:
            status, note = 'not_in_inherited_registry', 'Wave 1 coverage assertion omitted sections 0-19; no equivalence claimed'
        elif number in KNOWN_NON_EQUIVALENCES:
            status, note = KNOWN_NON_EQUIVALENCES[number]
        elif implementation and 'ModelAdapter' in implementation:
            status, note = 'requires_external_telemetry', 'interface only; no access to proprietary historical states is supplied'
        else:
            status, note = 'inherited_not_fully_audited', 'callable mapping retained; mathematical fidelity not established by index coverage'
        sections.append({'section':number,'inherited_reference':implementation,'status':status,'note':note})
    return {
        'format':'facemath.capability-audit.v3', 'mapping_is_mathematical_coverage_proof':False,
        'validation_scope':'captured_wave3_release_validation_not_runtime_self_introspection',
        'sections':sections,
        'wave3_active_instrument': [
            {'component':'strict observations and named anatomy','implementation':'observation.contracts / anatomy',
             'tests':'test_observation_contracts.py','status':'executed'},
            {'component':'local MediaPipe provider','implementation':'observation.provider.MediaPipeProvider',
             'tests':'test_wave3_provider.py','status':'mock_contract_executed_live_model_not_executed'},
            {'component':'symmetric robust registration and holdouts','implementation':'observation.canonical',
             'tests':'test_wave3_registration.py / test_wave3_instrument.py','status':'executed'},
            {'component':'piecewise-affine anatomy and finite-rotation strain','implementation':'observation.field',
             'tests':'test_wave3_field.py','status':'executed_against_analytic_controls'},
            {'component':'regional profile and conditional sensitivity','implementation':'observation.measure',
             'tests':'test_wave3_instrument.py','status':'executed_not_accuracy_calibrated'},
            {'component':'crossfade null and time-aware diagnostics','implementation':'observation.sequence_guard',
             'tests':'test_wave3_sequence.py','status':'executed'},
            {'component':'cohort compatibility leakage and dependence','implementation':'observation.cohort',
             'tests':'test_wave3_cohort.py','status':'executed_synthetic_tests_no_human_population_attached'},
            {'component':'raw artifact integrity and execution journal','implementation':'observation.artifacts / journal',
             'tests':'test_wave3_artifacts.py','status':'executed_not_external_attestation'},
        ],
    }
