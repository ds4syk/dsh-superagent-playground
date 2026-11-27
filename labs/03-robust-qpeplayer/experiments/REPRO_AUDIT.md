# P3 reproducibility audit

- Python 3.10.22 / NumPy 2.2.6 / SciPy 1.15.3
- git HEAD `c4cca045e8e0df9d061994eefbf58d796aff71a3`
- cwd `/mnt/storage1/syk/.superagent-work/three-paper-tracks-95244014fb3c/labs/03-robust-qpeplayer/experiments`
- overall: **PASS**

## Checks

| file | status |
|---|---|
| `synthetic/results/main_table_extended.json` | ok |
| `synthetic/results/gradient_table.json` | ok |
| `synthetic/results/e2e_table.json` | ok |
| `real/modelnet/results/main_table.json` | ok |
| `real/modelnet10/results/main_table.json` | ok |
| `real/modelnet10/results/meta.json` | ok |
| `real/threedmatch/results/threedmatch.json` | ok |
| `real/threedmatch/results/descriptor.json` | ok |
| `real/threedmatch/results/n_scaling.json` | ok |
| `real/threedmatch/results/verify_gt_direction.json` | ok |
| `real/threedmatch_extra/results/extra_scenes.json` | ok |
| `real/threedmatch_extra/results/verify_gt_direction_extra.json` | ok |
| `real/kitti/results/kitti.json` | ok |
| `real/kitti/results/c_sensitivity.json` | ok |
| `real/kitti/results/n_scaling.json` | ok |
| `real/kitti2/results/kitti2.json` | ok |
| `real/kitti2/results/verify_kitti2.json` | ok |
| `e2e_torch/results/e2e_torch.json` | ok |
| `e2e_torch/results/gradcheck.json` | ok |
| `inner/results/inner_solver.json` | ok |
| `systems/results/batch_gnc.json` | ok |
| `ablations/kernels/results/kernel_ablation.json` | ok |
| `ablations/gnc_schedule/results/gnc_schedule.json` | ok |
| `ablations/anisotropic/results/anisotropic.json` | ok |
| `ablations/breakdown/results/breakdown.json` | ok |
| `ablations/noise/results/noise.json` | ok |
| `ablations/ransac_lo/results/ransac_lo.json` | ok |
| `synthetic/results/main_table_extended_raw.jsonl` | ok (14500 lines) |
| `real/threedmatch/results/threedmatch_raw.jsonl` | ok (4800 lines) |
| `real/threedmatch/results/descriptor_raw.jsonl` | ok (600 lines) |
| `real/threedmatch/results/n_scaling_raw.jsonl` | ok (1280 lines) |
| `real/threedmatch_extra/results/extra_scenes_raw.jsonl` | ok (4800 lines) |
| `real/kitti/results/kitti_raw.jsonl` | ok (2400 lines) |
| `real/kitti/results/c_sensitivity_raw.jsonl` | ok (720 lines) |
| `systems/results/batch_gnc_raw.jsonl` | ok (800 lines) |
| `ablations/anisotropic/results/anisotropic_raw.jsonl` | ok (300 lines) |
| `ablations/ransac_lo/results/ransac_lo_raw.jsonl` | ok (4800 lines) |
| `real/kitti2/results/kitti2_raw.jsonl` | ok (2400 lines) |

## Large/cache files under experiments

- none

## SHA256 (key results)

- `synthetic/results/main_table_extended.json`: `720e184548227ddaa96e85372d1a1c2459b74865e7119ce9cbb478508514b9eb`
- `synthetic/results/gradient_table.json`: `667bfb7afbfddc29ecb26669876b02c9aa7471a61e6710ece05183799e680163`
- `synthetic/results/e2e_table.json`: `76236f3bbf4c12182667803f703d96eae96f0b784b3b73cec7b1f0aa83adaecf`
- `real/modelnet/results/main_table.json`: `aaae1bc95cef1f6b12b2411d41246758266ed3b3549b0fba3fdec7167a42567a`
- `real/modelnet10/results/main_table.json`: `e0fe247b84b2324b5e842afb5958a1d547289fa2995868bce0490f0fa63a0414`
- `real/modelnet10/results/meta.json`: `f2d75e74e964281bb2ca800eba73f571c164a55bab502b3e7ac880cfe99b553e`
- `real/threedmatch/results/threedmatch.json`: `db0e15fab745d99cccd84b1580b44172096b275aa8064a4ac1bc836355262988`
- `real/threedmatch/results/descriptor.json`: `45f65d98a5b709679fbcae18064f3d3b68c33bfaa437438574e0d759e0f05e87`
- `real/threedmatch/results/n_scaling.json`: `05bf5d2247579b726dc6f8c91b5c931e34038262d87253b4d98ea6c16ef092ee`
- `real/threedmatch/results/verify_gt_direction.json`: `0ba5a1eff97ef33692454b0c84ef46fa05d284a26c96fb10ce6528311a82e713`
- `real/threedmatch_extra/results/extra_scenes.json`: `d46a7be8a4ff8e3022fe2efcd71022504101de2c894c0d1759960229fc34f1d9`
- `real/threedmatch_extra/results/verify_gt_direction_extra.json`: `00caaca1995fbb2363e7c1136370613605e0e1138f2cc221f2903265aa1b430f`
- `real/kitti/results/kitti.json`: `9dc41f337eb10e498f16d50c4a69d5aa314cd5112966fe77ae9e12b0660f24b5`
- `real/kitti/results/c_sensitivity.json`: `25c1ef4b515a137550198141dfa5e9dcb6dfd8cb116e268fe8edafd009e36dfd`
- `real/kitti/results/n_scaling.json`: `cb34c1e925c1df9bf02e40595d95f38edde87f184400c42e2d17b1c7cef33a78`
- `real/kitti2/results/kitti2.json`: `60c534ce4367a582896c69b3cde9cfa0cb0e2a1be2f931fff709254bf685b817`
- `real/kitti2/results/verify_kitti2.json`: `21df7103c8568490abf82b1c3437bf257a5ff2c4d3f05514a4c7fb1e91fa4cb7`
- `e2e_torch/results/e2e_torch.json`: `ed9e3cf121f03803d2d296b25931c49995a7b4cc40a8c856458591e11f68d67d`
- `e2e_torch/results/gradcheck.json`: `de7b5aaa9b0f8438eb939e91c8a7d16b00cb4639bc480c693e7a10e709a3453b`
- `inner/results/inner_solver.json`: `8d9f20f2ebb31e7ad7ea3b1eb7113e2d9b40c6a32cd3ca5878be8140f6f020ba`
- `systems/results/batch_gnc.json`: `18da0c46e4c78c5b30e36283d4b02c22f9541af1cddab83da661d84ca6d763be`
- `ablations/kernels/results/kernel_ablation.json`: `ec63dc0a256d8aba89c3a50628c08a5bfe12760e33d86fd0e57c576bf106117a`
- `ablations/gnc_schedule/results/gnc_schedule.json`: `3aa2c17fce38eab91f06e0b99123f337c3768c8f7e1d790acad72e96873ae326`
- `ablations/anisotropic/results/anisotropic.json`: `0f7afb0b5b03c8955663b6da6eabe5b0da4a060ded22c2668661715046fb4700`
- `ablations/breakdown/results/breakdown.json`: `dde3b09811a348d2bc13940deb5049d8e6c1ad16801f67ad53dc98b59bae3095`
- `ablations/noise/results/noise.json`: `979aa63e96084013be11681447bead2c5f1f773abf38315b3941ef335982d4a2`
- `ablations/ransac_lo/results/ransac_lo.json`: `a077cf5787ececb1b48d9e8d12b7877e65a6683d02d93b9c87fed89a2610e7b0`
- `synthetic/results/main_table_extended_raw.jsonl`: `ae2172a522e275a1ea350d6d478d45c6c1af0a167b17ac415b493309d042e7f4`
- `real/threedmatch/results/threedmatch_raw.jsonl`: `fe1e5ca39296fb05d1f8113d0edb92c135dd819df4b638ee4608571b1a04a13c`
- `real/threedmatch/results/descriptor_raw.jsonl`: `56b00219a996bea4855319e8908aa5ff1c5af0bb1114739f876fa78ebf930317`
- `real/threedmatch/results/n_scaling_raw.jsonl`: `7a1353afb94de8178550bde0fd4c79948d136139d15e3526d7b7065d212ace0a`
- `real/threedmatch_extra/results/extra_scenes_raw.jsonl`: `dad4f6d564c5ab24997068e81437dbdc3fce62abdb3a58bf51eac221cdd010d6`
- `real/kitti/results/kitti_raw.jsonl`: `92593d13cf86aa5d5b313e3ab76e8536c3fe639bfeef7a4e249127f95f83ff39`
- `real/kitti/results/c_sensitivity_raw.jsonl`: `6d46758f95216a6f85f824a3c0c135dbc7d46c4d1745fea8e722d4bb454d2657`
- `systems/results/batch_gnc_raw.jsonl`: `f54668856b49446f5871b24da8dd44bacbfa290eb10940dfa24998e9a52cf30f`
- `ablations/anisotropic/results/anisotropic_raw.jsonl`: `ce1121b94188e184506a4d9c0e2f2730ef157b195bb4e19391bdfa7ac570fd0b`
- `ablations/ransac_lo/results/ransac_lo_raw.jsonl`: `ae696edc10eed3988dfcc57387767ac04ffd955868695475c877c7e3d93eff6b`
- `real/kitti2/results/kitti2_raw.jsonl`: `c5640abfa962b83aede49abbe7b8caae5450d50ce8c46b25595cda97e4fa53c8`
