"""Versioned assignments of native vertices to the shared anatomical grammar.

None means the historical implicit correspondence. It must never follow the
current provider default. A correspondence selects vertices; it does not fit,
move, average or replace detector coordinates.
"""
from types import MappingProxyType

from .contracts import ObservationError, object_digest

MP_REFERENCE8_V2 = "mediapipe478.reference8.v2"
SEMANTIC_21 = (
    "temple_left", "temple_right", "left_eye_outer", "left_eye_inner",
    "left_eye_top", "left_eye_bottom", "right_eye_inner", "right_eye_outer",
    "right_eye_top", "right_eye_bottom", "nose_bridge", "nose_tip",
    "left_nostril", "right_nostril", "mouth_left", "mouth_right",
    "upper_lip", "lower_lip", "chin", "jaw_left", "jaw_right",
)
MP_ANCHORS = MappingProxyType(dict(zip(SEMANTIC_21, (
    127, 356, 33, 133, 159, 145, 362, 263, 386, 374, 168, 1,
    98, 327, 61, 291, 0, 17, 152, 172, 397,
))))
MP_ANCHORS_V2 = MappingProxyType({**MP_ANCHORS, "nose_tip": 4})


def require_correspondence(schema: str, identity: str | None) -> None:
    if identity is not None and (schema != "mediapipe478.v1" or identity != MP_REFERENCE8_V2):
        raise ObservationError("unsupported landmark correspondence for observation schema")


def mediapipe_anchors(identity: str | None):
    require_correspondence("mediapipe478.v1", identity)
    return MP_ANCHORS if identity is None else MP_ANCHORS_V2


def correspondence_recipe(schema: str, identity: str | None) -> dict:
    """Preserve old recipe bytes; bind the full assignment for explicit mappings."""
    require_correspondence(schema, identity)
    if identity is None:
        return {}
    return {"correspondence_sha256": object_digest({
        "id": identity, "schema": schema, "anchors": dict(mediapipe_anchors(identity)),
        "sides": "viewer", "selection": "individual_native_vertices_no_coordinate_correction",
    })}
