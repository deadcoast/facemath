# Reference acquisition and evaluation protocol

## Before collecting labels

Define the acquisition domain: image type, resolution/eye-span range, frontal-pose limits, expression policy, illumination conditions and exclusion rules. Keep original rasters. A screen recording of a blended overlay is not a pair of independent source portraits.

Assign source images to declared acquisition groups before observing detector errors. All frames, crops or re-encodings of a source and all known repeated subjects/sessions belong in an appropriately dependent group/lineage. This is curator metadata, not automated identity recognition. Put entire groups into development, calibration or evaluation; never split repeated measurements randomly by image and call them independent.

Freeze model file digest, runtime version, perturbation recipe, scoring rule and minimum-rater policy after development. Do not choose error thresholds or remove difficult cases based on the final evaluation results.

## Labeling

Generate a separate blank workbench for each source image. The rater must not see the observer's point predictions during independent annotation. Use source anatomy definitions; viewer-side positions are literal image coordinates, with no hidden mirror normalization. Record unobservable points instead of guessed locations.

Use genuinely different raters when claiming multi-rater labels. A new ID alone does not establish an independent person. The interface clears previous labels when switching IDs, but cannot verify who is using it or what they have previously seen. The exported reference retains declared blinding, source digest, visibility and UI edit history. Local drafts are optional convenience, not the authoritative dataset.

The normalization currently requires all four eye corners in the reference consensus and at least an eight-pixel eye-midpoint span. A case lacking them cannot be scored under this protocol. Such cases need a separately prespecified acquisition accounting report, not an invented normalizer.

## Study manifest

Use `examples/validation.study.template.json` as a structural example, not a sufficient sample-size recommendation. Paths are local to the manifest directory. A case supports one of:

- `observation`: existing image-bound exported observer JSON, with no new inference claimed;
- `failure`: recorded attempt failure reason, retained in denominator;
- neither: invoke the concrete provider passed to `validate-observer --model ...`.

Do not supply both `observation` and `failure`. Each `references` list may contain several independent raters. Aggregate their records into one case rather than duplicating its image. Curator exclusions should be defined before model output is examined and accounted for outside the eligible-case metric denominator.

## Reading the result

`successful_case_mean_nme` is conditional; compare it with attempted/failed counts. PCK and unconditional AUC retain failed attempts. The eight-coordinate error maximum calibrates a simultaneous-reference score, not each landmark independently. Repeated group cases are reduced to a maximum, not treated as extra independent calibration samples.

A 90% nominal conformal envelope requires at least nine calibration groups even to have a finite corrected order statistic. Nine is an algebraic minimum, not a credible practical validation target. Missing/failed cases can make the envelope unbounded. Exchangeability, representativeness and a frozen protocol remain required; additional data alone cannot cure a mismatched domain.

Multi-rater disagreement is label variability, not detector covariance. A mean error to consensus can underestimate or overestimate error to actual anatomy. The nominal iBUG68 mapping must be manually reviewed for the chosen data source and point definitions, especially the chin and nose-tip conventions.

## Applying a calibration to a pair

Use the exact same observer recipe and acquisition domain. Supply query groups and all known source/session lineages. Source hashes and those declared identities are checked against the calibration set. A hash cannot detect an unreported re-encoded duplicate; the protocol needs honest curator metadata.

The eight-point fixed-gauge envelope is conservative and may be unresolved at the chosen tolerance. That is a real measurement outcome, not a reason to loosen the calibration or present the dense field as independently validated. Full-mesh confidence needs full-mesh validation or a justified and separately tested spatial error model.
