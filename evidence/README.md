# Evidence index

`VALIDATION.json` is the release summary generated from actual test XML, coverage JSON, source hashes, wheel-install output, replay output, and artifact verification.

**Current release:** `release-run/demo/` contains the numerical evidence and HTML report. `release-run/replay/` is its source-tree replay. `installed-wheel/replayed/` is its replay through the separately installed wheel. `release-run/command-*.json` and `installed-wheel/command-*.json` contain executed command arguments, working directories, timestamps, stdout, stderr, and return codes. They were recorded during those executions.

`pytest.xml`, `pytest-output.txt`, `coverage.json`, and `coverage.txt` record the coverage-instrumented test run. `release-run/pytest.xml` and `installed-wheel/pytest.xml` record two additional execution contexts. These are repeated runs of the same suite, not three times as many distinct tests.

`browser-qa-verified/` contains actual browser checks and screenshots. The managed browser blocked file-URL navigation, so the exact self-contained HTML was rendered through `set_content`; there were no remote requests or script errors. The initial blocked attempt is preserved as `browser-file-policy-failure.txt`.

`video_inspection/` and `video-frame-zero-roi-proposals.json` derive from the user's supplied recording. They are acquisition diagnostics, not a new facial-similarity estimate and not a public population dataset. The source-video hash and sample frame hashes are recorded. The original video is not redistributed as a test fixture.

**Development history:** `inherited-baseline.txt` records the 15-test Wave 2 baseline. `development/` retains intermediate runs from before final hardening; their source fingerprints differ from the release and exact replay under the final engine is not claimed. `environment-install.log` records failed optional-package acquisition. The initial coverage-startup failure is preserved; using a source-directory coverage target avoided the early OpenCV import problem. These failures are not counted as passing validation.

Hashes prove consistency with these artifacts, not an external attestation, trusted timestamp, or recovered proprietary internal inference. The live MediaPipe model test remains an explicit skip in this captured environment.
