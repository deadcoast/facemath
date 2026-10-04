---
version: alpha
name: FACEMATH evidence report
description: Local image-pair measurements with directly inspectable numerical evidence.
colors:
  background: "#10141a"
  panel: "#192029"
  ink: "#e7eef4"
  muted: "#a2b1c1"
  line: "#354252"
  primary: "#71ddd6"
typography:
  sans:
    fontFamily: "system-ui, sans-serif"
  mono:
    fontFamily: "ui-monospace, monospace"
rounded:
  DEFAULT: "0px"
spacing:
  page-max: "1320px"
  table-cell: "12px 14px"
components:
  table: {}
  disclosure: {}
  range: {}
---

# FACEMATH report

## Overview

This existing product surface is an instrument readout: image pairs, named
measurements, units and bounds in the same view. The intended reader compares
projected facial geometry on a desktop, with narrow-screen access preserved.
English is the current supported report language; no market-specific claims
are made. The original facial-mathematics document section 5 and
`docs/STRUCTURAL_COMPARISON_DESIGN.md` define the measurement scope.

Keep the established dark laboratory sheet and cyan image annotations. Avoid
decorative dashboards, identity scores, and status colors that replace text.
The runtime CSS in `src/facemath/observation/artifacts.py:_html` owns tokens;
this file records its values rather than generating a second theme.

## Colors

The frontmatter maps directly to `--bg`, `--panel`, `--ink`, `--muted`, `--line`
and `--accent`, respectively. Cyan marks links, annotations and keyboard focus.
Bounds and missingness are written in text. Scrollbar thumb/track/hover/active
map to muted/panel/ink/primary; forced colors use system colors.

## Typography

System sans renders labels and prose. Monospace renders hashes. Measurement
tables use tabular numerals; row labels and units stay visible together.

## Layout

The document owns vertical scrolling. Each wide table owns horizontal scrolling;
the fourteen-row structure table is finite and keyboard focusable. Desktop
content is capped at 1320px; below 760px, existing galleries stack and page
padding falls to 14px. Do not clip measurement columns to fit a phone.

## Elevation & Depth

Thin borders and panel tone group content. No shadows or floating overlays.

## Shapes

Square tables and image frames preserve the existing report language.

## Components

`readout.py` renders shared readouts; `artifacts.py` owns the report shell.
`intrinsic_report.py` is the existing sibling for detailed relation inspection.
Use semantic tables, native disclosures and the existing opacity range. These
reports contain no editing, remote request, authentication or destructive flow.
Unavailable rows remain visible; no loading state is invented for sealed data.

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Scrollbar | artifacts.py report CSS | Existing report shell | System colors in forced-colors mode | Browser narrow viewport and keyboard scroll |
| Read-only table | readout.py | Typed comparison profile | Nominal, calibrated, partial support | Structural tests and real native report |

## Do's and Don'ts

Display A, B, B-minus-A, units and reference bounds. Preserve observation values
when a model is attached. Keep missingness explicit. Never present dependent
relation counts as a resemblance probability or calibration as human acceptance.
