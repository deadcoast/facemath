# Primary API references and attribution

Consulted for Wave 3, September 11, 2026. These document external API contracts; they do not validate FACEMATH's scientific conclusions or certify its tests.

1. Google AI Edge / MediaPipe — **Face landmark detection guide for Python**.
   https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
   Used for FaceLandmarker options, `IMAGE` mode, `num_faces`, `detect`, normalized coordinates, 478 returned landmarks, optional blendshape results, and explicit local `model_asset_path`.

2. Google AI Edge / MediaPipe — **Face landmark detection overview**.
   https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
   Official task/model documentation. Model weights are externally supplied and not redistributed here.

3. OpenCV — **Optical Flow** (4.13.0 documentation available during development).
   https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html
   Used for Farnebäck grayscale image-flow conventions and forward sampling relationships. The method estimates image motion, not anatomical correctness across different people.

4. OpenCV — **Motion Analysis and Object Tracking**.
   https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html
   Function contract for `calcOpticalFlowFarneback`: paired single-channel images, dense two-component flow, parameter semantics.

5. NumPy / SciPy public numerical interfaces.
   https://numpy.org/doc/stable/reference/routines.linalg.html
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
   Implementation dependencies for SVD and Delaunay triangulation. Numeric runtime versions are recorded per run.

The new FACEMATH code is included as source for this project. Third-party libraries and downloaded model artifacts retain their own licenses. Google developer documentation identifies its sample-code licensing separately from model assets. No bundled file here claims ownership of those dependencies. The coarse ROI locator uses the Haar cascade shipped in the locally installed OpenCV package; its hash is recorded in proposals, and its weights are not copied into this release.
