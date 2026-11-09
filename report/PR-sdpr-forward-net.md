This is by an automated agent, not a human.

## What this is

A focused answer to the Feishu question: **in the SDPR paper/code, how is the differentiable
optimization layer's forward network wired, what network is it, and where does the dataset come
from?**

The short answer:

- **The SDPRLayer itself has no learnable parameters and is not a network.** In the paper's real
  experiment it sits at the end of a fully differentiable pipeline.
- The front end is **VGG16 (ImageNet-pretrained, truncated at `conv5_3`) + two UNet-style
  decoders** (keypoint logits, score map). It produces 768 keypoints, 960-d pixel descriptors
  and scores per image.
- The wiring is: stereo images → keypoints/descriptors/scores → dense descriptor matching →
  3D points + stereo covariance → scalar × matrix weights → **13-variable QCQP** →
  `SDPRLayer` (Mosek) → relative pose; the pose/keypoint losses backpropagate through the
  layer via implicit differentiation.
- The real-data experiment uses the **UTIAS In-The-Dark** subset of the UTIAS Long-Term
  Localization and Mapping Dataset (39 runs of a 260 m loop, hourly over ~30 h, Grizzly robot +
  Bumblebee XB3 stereo). The paper uses 100k train / 20k validation samples and tests on runs
  2/11/16/17/23/28/35. The simulated experiments generate their own data.

## Deliverables

- `report/sdpr_forward_network_dataset.md` — main report (Chinese), with tensor shapes,
  per-stage code references, dataset table, and reproduction caveats.
- `report/figures/pipeline.png` — original pipeline diagram.
- `deliverables/SDPR_前向网络与数据集_2026-11.docx` — Word version (default delivery).
- `kb/sdpr-forward-net/00.md`, `kb/sdpr-forward-net/01.md` — knowledge-base source pages for the Feishu reading copy.
- `src/` — diagram and DOCX build scripts.

## Verification

- Read the paper (`arXiv:2405.19309v3`, T-RO 2025) and the two code repositories.
- Code-verified: VGG16 layer truncation, 960-d descriptor concat, 16×16 / 768 keypoints,
  dense matching over 196,608 target pixels, 13×13 Q, 24 pose constraints, dataset splits,
  and the missing `pose_sdp_block.py` at `mat-weight-sdp-version` HEAD (`a9f0dc2`).
- DOCX rendered to PDF and inspected page by page (9 pages, Chinese fonts, table headers,
  figure, references).
- No numerical reproduction: In-The-Dark was not downloaded and the pipeline was not trained.

## Caveats

- Table VI/VII numbers are the paper's claims, not reproduced here.
- The public `mat-weight-sdp-version` branch does not run at HEAD: `pipeline.py` imports
  `pose_sdp_block.PoseSDPBlock`, but that file was never committed; the last usable commit is
  `ba66e408` (`src/model/sdpr_block.py`).
- The released `SDPPoseEstimator` defaults to `diff_qcqp=False` (SDPR-SDP), while the paper
  pipeline's `SDPRBlock` uses the `SDPRLayer` default `diff_qcqp=True` (SDPR-IS); align this
  switch before comparing gradients or timings.
