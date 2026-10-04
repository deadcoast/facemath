"""Conservative set arithmetic on one shared, eight-point anatomical observation.

A source is either product marginal ellipses or ONE full16 ellipsoid over
reference8. A product is an outer relaxation, not independent landmark errors.
All relations refer to the SAME eight variables. Bounds retain algebraic
cancellation in linear combinations and second-order bilinear terms.

This is floating-point conservative geometry with explicit roundoff padding,
not a formally machine-verified interval arithmetic implementation. See
WAVE6_MATHEMATICS.md for proofs, conditioning and non-equivalence limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .contracts import ObservationError, frozen_array

Array = NDArray[np.float64]
N_POINTS = 8
EYE_VECTOR = frozen_array([-.5, -.5, .5, .5, 0., 0., 0., 0.])
EPS = np.finfo(np.float64).eps


def pad(*values: float) -> float:
    """Small dimensionless roundoff budget; source charts are eye-normalized."""
    return 128 * EPS * max(1., *(abs(float(v)) for v in values))


@dataclass(frozen=True)
class Interval:
    lower: float = 0.
    upper: float = 0.
    available: bool = True
    reason: str = "bounded"

    def __post_init__(self) -> None:
        if (not np.isfinite([self.lower, self.upper]).all()
                or self.lower > self.upper or type(self.available) is not bool):
            raise ObservationError("invalid finite interval contract")
        if not self.available and (self.lower != 0 or self.upper != 0):
            raise ObservationError("unavailable interval must contain zero sentinels")

    @classmethod
    def missing(cls, reason: str) -> Interval:
        return cls(0., 0., False, reason)

    def to_dict(self) -> dict:
        return {"lower": self.lower if self.available else None,
                "upper": self.upper if self.available else None,
                "available": self.available, "reason": self.reason}


def outer(lower: float, upper: float, *, nonnegative: bool = False,
          reason: str = "bounded") -> Interval:
    if not np.isfinite([lower, upper]).all():
        return Interval.missing("nonfinite_arithmetic")
    slack = pad(lower, upper)
    lo = float(np.nextafter(lower - slack, -np.inf))
    hi = float(np.nextafter(upper + slack, np.inf))
    if nonnegative:
        lo, hi = max(0., lo), max(0., hi)
    return Interval(lo, hi, True, reason)


def divide(numerator: Interval, denominator: Interval) -> Interval:
    if not numerator.available or not denominator.available:
        return Interval.missing(numerator.reason if not numerator.available else denominator.reason)
    if denominator.lower <= 0:
        return Interval.missing("denominator_may_vanish")
    candidates = [a/b for a in (numerator.lower, numerator.upper)
                  for b in (denominator.lower, denominator.upper)]
    return outer(min(candidates), max(candidates))


def absolute_difference(a: Interval, b: Interval) -> Interval:
    if not a.available or not b.available:
        return Interval.missing(a.reason if not a.available else b.reason)
    lo = max(0., a.lower - b.upper, b.lower - a.upper)
    hi = max(abs(a.lower - b.upper), abs(a.upper - b.lower))
    return outer(lo, hi, nonnegative=True)


def coefficients(value: Array) -> Array:
    a = np.asarray(value, dtype=float)
    if a.shape != (N_POINTS,) or not np.isfinite(a).all():
        raise ObservationError("relation coefficients must be finite [8]")
    if abs(float(a.sum())) > 1e-12:
        raise ObservationError("intrinsic relation coefficients must remove translation")
    return a


def displacement(i: int, j: int) -> Array:
    if any(type(k) is not int or not 0 <= k < N_POINTS for k in (i, j)):
        raise ObservationError("invalid reference8 index")
    a = np.zeros(N_POINTS)
    a[i] += 1.
    a[j] -= 1.
    return a


@dataclass(frozen=True)
class EllipticSource:
    """Eight reference positions with product-marginal or full16 joint errors.

    Coordinates are in an observed source chart for numerical conditioning.
    That chart does not freeze the reference eye span: EYE_VECTOR remains an
    uncertain linear expression and all normalized relations divide by it.
    """

    centers: Array
    covariance: Array
    supported: NDArray[np.bool_]
    quantile: float
    directions: int = 128
    joint_covariance: Array | None = None
    joint_factor: Array | None = field(init=False, repr=False, default=None)
    normals: Array = field(init=False, repr=False)
    radii: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        p, c = frozen_array(self.centers), frozen_array(self.covariance)
        mask = np.asarray(self.supported)
        if p.shape != (N_POINTS, 2) or c.shape != (N_POINTS, 2, 2):
            raise ObservationError("elliptic source requires [8,2] centers and [8,2,2] covariance")
        if mask.shape != (N_POINTS,) or mask.dtype != np.bool_:
            raise ObservationError("source support must be explicit boolean [8]")
        if not np.allclose(c, c.swapaxes(-1, -2), rtol=1e-12, atol=1e-15):
            raise ObservationError("source covariance must be symmetric")
        eig = np.linalg.eigvalsh(c)
        if np.any(eig < -1e-13 * np.maximum(1., np.max(np.abs(c), axis=(1, 2)))[:, None]):
            raise ObservationError("source covariance must be positive semidefinite")
        if not np.isfinite(self.quantile) or self.quantile < 0:
            raise ObservationError("finite nonnegative error multiplier required")
        if type(self.directions) is not int or self.directions < 32 or self.directions > 2048 or self.directions % 2:
            raise ObservationError("support directions must be an even integer in [32,2048]")
        if self.joint_covariance is not None:
            joint = frozen_array(self.joint_covariance)
            if joint.shape != (16, 16) or not np.allclose(joint, joint.T, rtol=1e-12, atol=1e-15):
                raise ObservationError("joint source requires symmetric full16 covariance")
            eigenvalues, vectors = np.linalg.eigh(joint)
            if eigenvalues.min() < -1e-12 * max(1., float(np.max(np.abs(joint)))):
                raise ObservationError("joint source covariance must be positive semidefinite")
            blocks = joint.reshape(8, 2, 8, 2)
            marginal = np.stack([blocks[i, :, i, :] for i in range(8)])
            if not np.allclose(marginal, c, rtol=1e-11, atol=1e-14):
                raise ObservationError("joint source marginal covariance differs from source blocks")
            factor = (vectors * np.sqrt(np.maximum(0., eigenvalues)))
            object.__setattr__(self, "joint_covariance", joint)
            object.__setattr__(self, "joint_factor", frozen_array(factor.reshape(8, 2, 16)))
        theta = np.arange(self.directions) * (2*pi/self.directions)
        object.__setattr__(self, "centers", p)
        object.__setattr__(self, "covariance", c)
        object.__setattr__(self, "supported", frozen_array(mask, bool))
        object.__setattr__(self, "normals", frozen_array(np.column_stack((np.cos(theta), np.sin(theta)))))
        object.__setattr__(self, "radii", frozen_array(self.quantile * np.sqrt(np.maximum(eig[:, -1], 0.))))


class SourceRelations:
    """One cached linear-expression graph shared by all relation families.

    This short-lived evaluator owns caches, while exported source/result state
    remains immutable. No metric reacquires coordinates, realigns anatomy or
    estimates its own covariance.
    """

    def __init__(self, source: EllipticSource) -> None:
        self.source = source
        self._norm_cache: dict[tuple[float, ...], Interval] = {}
        self._radial_cache: dict[tuple[float, ...], float] = {}
        self._marginal = (SourceRelations(EllipticSource(source.centers, source.covariance,
                             source.supported, source.quantile, source.directions))
                          if source.joint_covariance is not None else None)
        self._projected_radius = source.quantile * np.sqrt(np.maximum(0., np.einsum(
            "mi,nij,mj->mn", source.normals, source.covariance, source.normals)))

    def combination_factor(self, a: Array) -> Array:
        """2x16 map of ONE shared latent error; never concatenate eight independent balls."""
        if self.source.joint_factor is None:
            raise ObservationError("full-shape combination requires a joint source")
        return np.einsum("i,ijk->jk", coefficients(a), self.source.joint_factor)

    def joint_support(self, a: Array) -> Array:
        factor = self.combination_factor(a)
        return self.source.quantile * np.linalg.norm(self.source.normals @ factor, axis=1)

    def observable(self, a: Array) -> bool:
        a = coefficients(a)
        return bool(np.all(self.source.supported[a != 0]))

    def nominal_vector(self, a: Array) -> Array:
        return coefficients(a) @ self.source.centers

    def radius(self, a: Array) -> float:
        """Outer radius of an error combination, including shared-point cancellation."""
        a = coefficients(a)
        key = tuple(a)
        if key not in self._radial_cache:
            if self._marginal is not None:
                radius = self.source.quantile * float(np.linalg.norm(self.combination_factor(a), 2))
                self._radial_cache[key] = min(radius+pad(radius), self._marginal.radius(a))
                return self._radial_cache[key]
            h = self._projected_radius @ np.abs(a)
            radial = float(np.abs(a) @ self.source.radii)
            angular = float(h.max()) / cos(pi/self.source.directions)
            self._radial_cache[key] = min(radial, angular) + pad(radial, angular)
        return self._radial_cache[key]

    def norm(self, a: Array) -> Interval:
        a = coefficients(a)
        key = tuple(a)
        if key in self._norm_cache:
            return self._norm_cache[key]
        if not self.observable(a):
            result = Interval.missing("unobserved_dependency")
        elif not np.any(a):
            result = Interval(0., 0., True, "algebraically_zero")
        else:
            center = a @ self.source.centers
            nominal = float(np.linalg.norm(center))
            support = (self.joint_support(a) if self._marginal is not None
                       else self._projected_radius @ np.abs(a))
            projections = self.source.normals @ center
            radial = self.radius(a) if self._marginal is not None else float(np.abs(a) @ self.source.radii)
            lower = max(0., nominal-radial, float(np.max(projections-support)))
            upper = min(nominal+radial, float(np.max(projections+support))/cos(pi/self.source.directions))
            result = outer(lower, upper, nonnegative=True)
            if self._marginal is not None:
                relaxed = self._marginal.norm(a)
                result = Interval(max(result.lower, relaxed.lower), min(result.upper, relaxed.upper))
        self._norm_cache[key] = result
        self._norm_cache[tuple(-a)] = result
        return result

    @property
    def eye_span(self) -> Interval:
        return self.norm(EYE_VECTOR)

    def length_ratio(self, a: Array, denominator: Array = EYE_VECTOR) -> Interval:
        a, denominator = coefficients(a), coefficients(denominator)
        divisor = self.norm(denominator)
        if (self.observable(a) and divisor.available and divisor.lower > 0
                and (np.array_equal(a, denominator) or np.array_equal(a, -denominator))):
            return Interval(1., 1., True, "shared_expression_cancels")
        result = divide(self.norm(a), divisor)
        return Interval(max(0., result.lower), max(0., result.upper), result.available, result.reason)

    def bilinear(self, a: Array, b: Array, kind: Literal["dot", "cross"]) -> Interval:
        a, b = coefficients(a), coefficients(b)
        if kind not in ("dot", "cross"):
            raise ObservationError("bilinear kind must be dot or cross")
        if not self.observable(a) or not self.observable(b):
            return Interval.missing("unobserved_dependency")
        u, v = a @ self.source.centers, b @ self.source.centers
        if self._marginal is not None:
            if kind == "cross" and (np.array_equal(a, b) or np.array_equal(a, -b)):
                return Interval(0., 0., True, "shared_expression_cancels")
            # (u+q Az)^T M (v+q Bz) with a SINGLE ||z||<=1.
            # The linear term is one support norm; quadratic eigenvalue bounds
            # retain correlations and repeated-point cancellation.
            matrix = np.eye(2) if kind == "dot" else np.array([[0., 1.], [-1., 0.]])
            fa, fb = self.combination_factor(a), self.combination_factor(b)
            linear = fa.T @ matrix @ v + fb.T @ matrix.T @ u
            quadratic = fa.T @ matrix @ fb
            eigenvalues = np.linalg.eigvalsh((quadratic+quadratic.T)/2)
            q = self.source.quantile
            center = float(u @ matrix @ v)
            budget = q*float(np.linalg.norm(linear))
            joint = outer(center-budget+q*q*min(0., float(eigenvalues[0])),
                          center+budget+q*q*max(0., float(eigenvalues[-1])))
            relaxed = self._marginal.bilinear(a, b, kind)
            return Interval(max(joint.lower, relaxed.lower), min(joint.upper, relaxed.upper))
        if kind == "cross":
            center = float(u[0]*v[1] - u[1]*v[0])
            grad_u, grad_v = np.array([v[1], -v[0]]), np.array([-u[1], u[0]])
        else:
            center = float(u @ v)
            grad_u, grad_v = v, u
        gradient = a[:, None]*grad_u + b[:, None]*grad_v
        linear_radius = self.source.quantile * float(np.sqrt(np.maximum(0., np.einsum(
            "ni,nij,nj->n", gradient, self.source.covariance, gradient))).sum())
        lo2, hi2 = 0., 0.
        r = self.source.radii
        if kind == "dot":
            diagonal = a*b*r*r
            lo2, hi2 = float(np.minimum(0., diagonal).sum()), float(np.maximum(0., diagonal).sum())
        for i in range(N_POINTS):
            for j in range(i+1, N_POINTS):
                coefficient = a[i]*b[j] - a[j]*b[i] if kind == "cross" else a[i]*b[j] + a[j]*b[i]
                residual = abs(coefficient)*r[i]*r[j]
                lo2 -= residual
                hi2 += residual
        # In a cross product, repeated point perturbations cancel exactly:
        # cross(delta_i, delta_i)=0. Do not treat the shared apex as two errors.
        if kind == "cross" and (np.array_equal(a, b) or np.array_equal(a, -b)):
            return Interval(0., 0., True, "shared_expression_cancels")
        return outer(center-linear_radius+lo2, center+linear_radius+hi2)

    def angle(self, a: Array, b: Array) -> Interval:
        a, b = coefficients(a), coefficients(b)
        na, nb = self.norm(a), self.norm(b)
        if not na.available or not nb.available:
            return Interval.missing("unobserved_dependency")
        if min(na.lower, nb.lower) <= 0:
            return Interval.missing("angle_arm_may_vanish")
        if np.array_equal(a, b):
            return Interval(0., 0., True, "shared_expression_cancels")
        if np.array_equal(a, -b):
            return Interval(pi, pi, True, "shared_expression_cancels")
        dot = self.bilinear(a, b, "dot")
        # Independent outer constraint from the law of cosines; intersect only
        # valid outer bounds, never assume independent side-length errors.
        nc = self.norm(a-b)
        cosine_lo = .5*(na.lower**2 + nb.lower**2 - nc.upper**2)
        cosine_hi = .5*(na.upper**2 + nb.upper**2 - nc.lower**2)
        dot = Interval(max(dot.lower, cosine_lo-pad(cosine_lo)),
                       min(dot.upper, cosine_hi+pad(cosine_hi)))
        denominator = outer(na.lower*nb.lower, na.upper*nb.upper, nonnegative=True)
        c = divide(dot, denominator)
        if not c.available:
            return Interval.missing(c.reason)
        lo = float(np.arccos(np.clip(c.upper, -1., 1.)))
        hi = float(np.arccos(np.clip(c.lower, -1., 1.)))
        result = outer(lo, hi)
        return Interval(max(0., result.lower), min(pi, result.upper))

    def signed_area_ratio(self, a: Array, b: Array) -> Interval:
        """Signed DOUBLE triangle area / uncertain reference eye span squared."""
        d = self.eye_span
        if not d.available:
            return Interval.missing(d.reason)
        if d.lower <= 0:
            return Interval.missing("denominator_may_vanish")
        denominator = outer(d.lower**2, d.upper**2, nonnegative=True)
        return divide(self.bilinear(a, b, "cross"), denominator)

    def projected_ratio(self, a: Array, *, perpendicular: bool = False) -> Interval:
        """Signed eye-axis projection / uncertain eye span, on this same graph."""
        a = coefficients(a)
        d = self.eye_span
        if not d.available:
            return Interval.missing(d.reason)
        if d.lower <= 0:
            return Interval.missing("denominator_may_vanish")
        if not self.observable(a):
            return Interval.missing("unobserved_dependency")
        if not np.any(a):
            return Interval(0., 0., True, "algebraically_zero")
        if np.array_equal(a, EYE_VECTOR) or np.array_equal(a, -EYE_VECTOR):
            value = 0. if perpendicular else (1. if np.array_equal(a, EYE_VECTOR) else -1.)
            return Interval(value, value, True, "shared_expression_cancels")
        denominator = outer(d.lower**2, d.upper**2, nonnegative=True)
        return divide(self.bilinear(EYE_VECTOR, a, "cross" if perpendicular else "dot"), denominator)

    def asymmetry(self, a: Array, b: Array) -> Interval:
        """Signed (left_length-right_length)/(left_length+right_length)."""
        la, lb = self.norm(a), self.norm(b)
        if not la.available or not lb.available:
            return Interval.missing("unobserved_dependency")
        if la.lower + lb.lower <= 0:
            return Interval.missing("asymmetry_denominator_may_vanish")
        if np.array_equal(a, b) or np.array_equal(a, -b):
            return Interval(0., 0., True, "shared_expression_cancels")
        # The contrast increases in la and decreases in lb, on nonnegative lengths.
        lower = (la.lower-lb.upper)/(la.lower+lb.upper)
        upper = (la.upper-lb.lower)/(la.upper+lb.lower)
        result = outer(lower, upper)
        return Interval(max(-1., result.lower), min(1., result.upper))
