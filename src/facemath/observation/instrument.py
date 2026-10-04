"""The Wave 3 orchestration boundary: one state, one field, one measured profile."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .anatomy import anatomy_for
from .canonical import CanonicalState, InstrumentConfig, canonicalize
from .contracts import LandmarkProvider, Observation, ObservationError, array_digest, rgb_image
from .field import AnatomicalField, FlowDiagnostics, build_field, flow_diagnostics
from .journal import Journal, environment
from .measure import MeasurementProfile, measure
from .edge_support import EdgeDiagnostics, diagnose_edges
from .error_model import LocalizationCalibration, LocalizationEnvelope, assess_pair
from .bias import BiasCalibration, BiasEnvelope, assess_bias_pair
from .intrinsic import IntrinsicProfile, measure_intrinsic
from .acquisition import MeasuredGroup


@dataclass(frozen=True)
class InstrumentResult:
    state: CanonicalState
    field: AnatomicalField
    flow: FlowDiagnostics | None
    profile: MeasurementProfile
    journal_jsonl: str
    environment: dict
    edges: EdgeDiagnostics | None = None
    localization: LocalizationEnvelope | BiasEnvelope | None = None
    intrinsic: IntrinsicProfile | None = None
    acquisition_groups: tuple[MeasuredGroup, MeasuredGroup] | None = None
    qualification_evidence: object | None = None

    def evidence(self) -> dict:
        return {
            "format": "facemath.evidence.v3", "recipe_sha256": self.state.recipe_sha256,
            "observation_a_sha256": self.state.observation_a.digest,
            "observation_b_sha256": self.state.observation_b.digest,
            "configuration": self.state.config.recipe(),
            "normalization": {"kind": "geometric_mean_of_registered_interocular_spans",
                              "canonical_pixels": self.state.normalization_span_px},
            "measurement_profile": [m.to_dict() for m in self.profile.measurements],
            "regional_comparison": self.profile.regional.to_dict(),
            "registration_diagnostics": {
                "heldout": [{"region": name, "available": bool(available), "failure": failure}
                    for name, available, failure in zip(self.profile.heldout_labels,
                        self.profile.heldout_fit_supported, self.profile.heldout_failures)],
                "sensitivity": {"requested_draws": len(self.profile.sensitivity_draws),
                    "completed_draws": int(self.profile.sensitivity_draw_supported.sum()),
                    "failures": [{"draw": i, "reason": reason} for i, reason in self.profile.sensitivity_failures],
                    "intervals_available": bool(self.profile.sensitivity_draw_supported.all()),
                    "incomplete_experiment_rule": "retain all slots; withhold intervals; no successful-draw-only quantiles"}},
            "warnings": list(self.profile.warnings),
            "consensus_similarity_percent": None,
            "population_percentile": None,
            "edge_diagnostics": self.edges.summary() if self.edges is not None else None,
            "localization_envelope": self.localization.to_dict() if self.localization is not None else None,
            "intrinsic_profile": self.intrinsic.to_dict() if self.intrinsic is not None else None,
            "empirical_admission": self.qualification_evidence.summary if self.qualification_evidence is not None else {"status": "not_requested", "passed": False, "scope": "research analysis; empirical acceptance not asserted"},
            "measured_acquisition": [g.to_dict() for g in self.acquisition_groups] if self.acquisition_groups is not None else None,
            "limits": [
                "2D projected geometry, not identity recognition or proof of 3D shape",
                "model contours and interpolated fields derive from the same landmarks; not independent evidence",
                "sensitivity intervals are conditional perturbation intervals, not measured localization-accuracy confidence intervals",
                "flow consistency tests image matching, not anatomical correctness across people",
                "within-tolerance fractions have an explicit threshold and denominator; they are not resemblance probabilities",
                "no population rarity without an exact-recipe held-out reference cohort",
                "the report contains new tool-side evidence, not inaccessible historical model activations",
            ],
        }

    def arrays(self) -> dict[str, np.ndarray]:
        state, field, profile = self.state, self.field, self.profile
        arrays = {
            "observation_a_xy": state.observation_a.xy, "observation_b_xy": state.observation_b.xy,
            "landmark_supported": state.observation_a.in_bounds & state.observation_b.in_bounds,
            "sensitivity_region_supported": np.array([
                np.sum((state.observation_a.in_bounds & state.observation_b.in_bounds)[state.anatomy.region_indices(r)]) >= 2
                for r in profile.sensitivity_regions
            ]),
            "observation_a_samples": state.observation_a.samples, "observation_b_samples": state.observation_b.samples,
            "observation_a_covariance": state.observation_a.covariance, "observation_b_covariance": state.observation_b.covariance,
            "canonical_a_xy": state.xy_a, "canonical_b_xy": state.xy_b,
            "source_to_canonical_a": state.source_to_canonical_a, "source_to_canonical_b": state.source_to_canonical_b,
            "canonical_to_source_a": np.linalg.inv(state.source_to_canonical_a), "canonical_to_source_b": np.linalg.inv(state.source_to_canonical_b),
            "source_support_a": state.valid_source_a, "source_support_b": state.valid_source_b,
            "photometric_support": state.photometric_support, "registration_weights": state.fit_weights,
            "anatomical_uv": field.uv, "triangle_indices": field.triangles, "simplex_index": field.simplex_index,
            "triangle_jacobians": field.jacobians, "triangle_determinants": field.determinant,
            "triangle_green_strain": field.green_strain, "triangle_principal_log_stretch": field.principal_log_stretch,
            "triangle_condition": field.condition, "triangle_valid": field.valid_triangles,
            "anatomical_support": field.valid, "landmark_residual_vectors": profile.residual_vectors,
            "relational_distortion": profile.relational_distortion,
            "heldout_residual_vectors": profile.heldout_residuals, "sensitivity_draws": profile.sensitivity_draws,
            "heldout_fit_supported": profile.heldout_fit_supported,
            "sensitivity_draw_supported": profile.sensitivity_draw_supported,
        }
        arrays.update(profile.regional.arrays())
        arrays.update({f"region_a_{k}": v for k, v in state.masks_a.items()})
        arrays.update({f"region_b_{k}": v for k, v in state.masks_b.items()})
        if self.flow is not None:
            arrays.update({f"flow_{name}": getattr(self.flow, name) for name in self.flow.__dataclass_fields__})
        if self.edges is not None:
            arrays.update(self.edges.arrays())
        if self.localization is not None:
            arrays.update(self.localization.arrays())
            if self.localization.directional is not None:
                arrays.update(self.localization.directional.arrays())
        if self.intrinsic is not None:
            arrays.update(self.intrinsic.arrays())
        if self.acquisition_groups is not None:
            for name, group in zip(("a", "b"), self.acquisition_groups):
                arrays.update(group.arrays(f"acquisition_{name}_"))
        return arrays


    def raw_geometry_arrays(self) -> dict[str, np.ndarray]:
        """Arrays that adding localization qualification is forbidden to change.

        New error-set centers, bounds and certificates are explicitly excluded;
        the original diagnostic state and new observed relations are retained.
        """
        intrinsic_observed = {
            "intrinsic_raw_points", "intrinsic_source_covariance", "intrinsic_source_chart",
            "intrinsic_source_supported", "intrinsic_nominal", "intrinsic_nominal_valid",
            "intrinsic_relation_a", "intrinsic_relation_b", "intrinsic_support_normals",
        }
        structural_observed = {"structure_nominal", "structure_nominal_valid",
                               "structure_expression_a", "structure_expression_b"}
        return {key: value for key, value in self.arrays().items()
                if not key.startswith(("localization_", "acquisition_"))
                and (not key.startswith("intrinsic_") or key in intrinsic_observed)
                and (not key.startswith("structure_") or key in structural_observed)}


class InstrumentFailure(ObservationError):
    """An expected acquisition/geometry failure with its actual partial journal."""

    def __init__(self, message: str, journal_jsonl: str) -> None:
        super().__init__(message)
        self.journal_jsonl = journal_jsonl


class FacialInstrument:
    def __init__(self, config: InstrumentConfig | None = None, provider: LandmarkProvider | None = None,
                 *, localization_calibration: LocalizationCalibration | BiasCalibration | None = None,
                 query_groups: tuple[str, str] = ("", ""), domain: str = "",
                 query_lineages: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
                 query_strata: tuple[str, str] = ("", ""),
                 acquisition_groups: tuple[MeasuredGroup, MeasuredGroup] | None = None,
                 qualification_evidence=None) -> None:
        self.config = config or InstrumentConfig()
        self.provider = provider
        self.localization_calibration = localization_calibration
        self.query_groups = tuple(query_groups)
        self.query_lineages = tuple(tuple(x) for x in query_lineages)
        self.domain = domain
        self.query_strata = tuple(query_strata)
        self.acquisition_groups = acquisition_groups
        self.qualification_evidence = qualification_evidence
        if qualification_evidence is not None and self.localization_calibration is None:
            self.localization_calibration = qualification_evidence.calibration

    def run(self, image_a: np.ndarray, image_b: np.ndarray,
            observation_a: Observation | None = None,
            observation_b: Observation | None = None) -> InstrumentResult:
        journal = Journal()
        try:
            return self._execute(image_a, image_b, observation_a, observation_b, journal)
        except ObservationError as exc:
            raise InstrumentFailure(str(exc), journal.jsonl()) from exc

    def _execute(self, image_a: np.ndarray, image_b: np.ndarray,
                 observation_a: Observation | None, observation_b: Observation | None,
                 journal: Journal) -> InstrumentResult:
        with journal.step("inputs.validate", {}) as output:
            a, b = rgb_image(image_a), rgb_image(image_b)
            output.update(a_sha256=array_digest(a), b_sha256=array_digest(b), shape_a=list(a.shape), shape_b=list(b.shape))
        if self.qualification_evidence is not None:
            with journal.step("admission.require_recomputed_heldout_criteria", self.qualification_evidence.summary) as output:
                self.qualification_evidence.require_admitted()
                output.update(passed=True, observer_reexecuted=False)
        from .stratified import StratifiedJointCalibration, BoundStratifiedCalibration
        calibration = self.localization_calibration
        model = calibration.model if isinstance(calibration, BoundStratifiedCalibration) else calibration
        measured_policy = getattr(getattr(model, "policy", None), "measurement_policy", None)
        groups, lineages, strata = self.query_groups, self.query_lineages, self.query_strata
        acquisition = self.acquisition_groups
        if measured_policy is not None:
            with journal.step("acquisition.measure_original_sources_before_observer", {
                "policy_sha256": measured_policy.digest, "uses_predictions_or_reference_labels": False}) as output:
                acquisition = acquisition or getattr(calibration, "measured_groups", None)
                if acquisition is None or len(acquisition) != 2:
                    raise ObservationError("measured calibration requires two raw source acquisition groups")
                acquisition = tuple(g.recompute() for g in acquisition)
                if any(g.policy.digest != measured_policy.digest for g in acquisition):
                    raise ObservationError("source measurement policy differs from calibrated policy")
                for image, group in zip((a, b), acquisition):
                    group.require_source(image)
                if groups == ("", ""):
                    groups = tuple(g.group_id for g in acquisition)
                if lineages == ((), ()):
                    lineages = tuple(g.lineage_ids for g in acquisition)
                derived = tuple(g.category for g in acquisition)
                output.update(groups=[g.to_dict() for g in acquisition], group_digests=[g.digest for g in acquisition])
                if strata != ("", "") and strata != derived:
                    raise ObservationError("declared category conflicts with measured source pixels; no manual override")
                strata = derived
                if groups != tuple(g.group_id for g in acquisition) or lineages != tuple(g.lineage_ids for g in acquisition):
                    raise ObservationError("query context must retain the complete measured group and lineage inventory")
                if groups[0] == groups[1] and acquisition[0].digest != acquisition[1].digest:
                    raise ObservationError("same query group cannot use different measured source inventories")
                for group in acquisition:
                    if model.radius(group.category).quantile is None:
                        raise ObservationError(f"measured category {group.category!r} is unsupported: {model.radius(group.category).status}; no fallback")
                    for source in group.sources:
                        if (source.image_sha256 in model.residual.calibration_images or group.group_id in dict(model.residual.group_scores)
                                or set(source.lineage_ids) & set(model.residual.calibration_lineages)):
                            raise ObservationError("measured group context leaks calibration evidence")
                        model.bias.require_disjoint(source.image_sha256, group.group_id, source.lineage_ids)
                # A serialized route is a claim until its raw group is remeasured.
                if isinstance(calibration, BoundStratifiedCalibration):
                    for route, group in zip(calibration.routes, acquisition):
                        if route.measured_group_sha256 != group.digest:
                            raise ObservationError("saved measured route differs from supplied original source group")
                    calibration = model
        elif acquisition is not None:
            raise ObservationError("source groups require a measured calibration; use acquisition-group for standalone diagnostics")
        if self.qualification_evidence is not None:
            with journal.step("admission.require_disjoint_query_sources", {}) as output:
                self.qualification_evidence.require_query(acquisition, calibration)
                output.update(evaluation_sources_excluded=True, scope=self.qualification_evidence.summary["scope"])
        with journal.step("observation.acquire", {"provided_a": observation_a is not None, "provided_b": observation_b is not None}) as output:
            if observation_a is None or observation_b is None:
                if self.provider is None:
                    raise ObservationError("supply two image-bound observations or a concrete local landmark provider")
                observation_a = observation_a or self.provider.observe(a)
                observation_b = observation_b or self.provider.observe(b)
            anatomy = anatomy_for(observation_a)
            anatomy_for(observation_b)
            output.update(observation_a_sha256=observation_a.digest, observation_b_sha256=observation_b.digest,
                          provider_a=observation_a.provider, provider_b=observation_b.provider,
                          samples_a=len(observation_a.samples), samples_b=len(observation_b.samples))
        if isinstance(calibration, StratifiedJointCalibration):
            with journal.step("calibration.route_acquisition_sources", {
                "model_sha256": calibration.digest, "declared_strata": strata,
                "routing_uses_reference_or_pair_error": False}) as output:
                calibration = calibration.bind((observation_a, observation_b), groups,
                    strata, self.domain, lineages, measured_groups=acquisition)
                output.update(calibration.routing_evidence())
        elif isinstance(calibration, BoundStratifiedCalibration):
            expected_strata = tuple(r.stratum for r in calibration.routes)
            if strata not in (("", ""), expected_strata):
                raise ObservationError("query strata conflict with bound acquisition routes")
            with journal.step("calibration.route_acquisition_sources", {
                "model_sha256": calibration.model.digest, "declared_strata": expected_strata,
                "routing_uses_reference_or_pair_error": False}) as output:
                for observation, group, ids in zip((observation_a, observation_b), groups, lineages):
                    calibration.require_query(observation, group, self.domain, ids)
                output.update(calibration.routing_evidence())
        elif strata != ("", ""):
            if (not isinstance(calibration, BoundStratifiedCalibration)
                    or strata != tuple(r.stratum for r in calibration.routes)):
                raise ObservationError("query strata require a matching category-conditioned calibration")
        with journal.step("canonical.shared_state", self.config.recipe()) as output:
            state = canonicalize(a, b, observation_a, observation_b, anatomy, self.config)
            output.update(recipe_sha256=state.recipe_sha256, a_matrix=state.source_to_canonical_a.tolist(),
                          b_matrix=state.source_to_canonical_b.tolist(), fit_anchor_names=list(state.fit_names),
                          fit_weights=state.fit_weights.tolist())
        with journal.step("field.anatomical_interpolation", {"method": "piecewise_affine_delaunay", "direction": "A_to_B"}) as output:
            field = build_field(state)
            output.update(triangles=len(field.triangles), valid_triangles=int(field.valid_triangles.sum()),
                          supported_pixels=int(field.valid.sum()), folded_fraction=field.folded_fraction,
                          uv_sha256=array_digest(field.uv))
        flow = None
        if self.config.enable_flow:
            with journal.step("field.independent_flow_diagnostic", {"method": "uint8_farneback_forward_backward"}) as output:
                flow = flow_diagnostics(state, field)
                output.update(admissible_pixels=int(flow.admissible.sum()), concordant_pixels=int(flow.concordant.sum()),
                              uv_sha256=array_digest(flow.uv))
        with journal.step("measure.shared_state_profile", {"uncertainty_draws": self.config.uncertainty_draws, "seed": self.config.seed}) as output:
            profile = measure(state, field, flow)
            output.update(measurements=len(profile.measurements),
                          withheld=sum(m.value is None for m in profile.measurements),
                          sensitivity_sha256=array_digest(profile.sensitivity_draws))
        edges = None
        if self.config.enable_edges:
            with journal.step("observation.image_edge_support", {"landmark_guided": True, "moves_landmarks": False}) as output:
                edges = diagnose_edges(state)
                output.update(edges.summary())
        localization = None
        if calibration is not None:
            with journal.step("validation.localization_envelope", {"calibration_sha256": calibration.digest}) as output:
                assessor = assess_bias_pair if isinstance(calibration, BiasCalibration) else assess_pair
                localization = assessor(state, calibration,
                                          query_groups=groups, domain=self.domain, query_lineages=lineages)
                output.update(supported_points=int(localization.supported.sum()),
                              kind="fixed_observed_gauge_error_envelope_not_shape_confidence")
        intrinsic = None
        if self.config.enable_intrinsic:
            with journal.step("measure.intrinsic_reference_shapes", {
                "reference_points": 8, "changes_registration": False,
                "uncertain_normalization": True,
            }) as output:
                intrinsic = measure_intrinsic(state, localization)
                output.update(relations=len(intrinsic.relations),
                              bounded_relations=int(intrinsic.gap_available.sum()),
                              error_set="joint_full16" if intrinsic.joint is not None else "marginal_product",
                              shape_bound_available=intrinsic.shape.available,
                              recipe_sha256=intrinsic.recipe_sha256)
        env = environment()
        journal.record("execution.complete", {"environment": env, "recipe_sha256": state.recipe_sha256})
        return InstrumentResult(state, field, flow, profile, journal.jsonl(), env, edges, localization, intrinsic, acquisition, self.qualification_evidence)
