---
task_id: witness-state-threshold
role: Vision Algorithm Researcher
objective: Define a defensible state metric and provisional thresholds for anti-loosening witness marks
status: complete
confidence: medium
sources_found: 10
acceptance_met: yes
---

## Sources

[1] Vision-Based Real-Time Bolt Loosening Detection by Identifying Anti-Loosening Lines | https://pmc.ncbi.nlm.nih.gov/articles/PMC11511543/ | peer-reviewed primary paper | 2024
[2] Detection of loosening angle for mark bolted joints with computer vision and geometric imaging | https://www.sciencedirect.com/science/article/pii/S0926580522003909 | peer-reviewed primary paper | 2022
[3] CN113469966B, Train bolt looseness detection method based on anti-loosening line identification | https://patents.google.com/patent/CN113469966B/en | primary patent, not independent validation | 2023
[4] 钢结构工程高强度螺栓微小松动视觉检测方法研究 | https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2024.02.011 | peer-reviewed primary paper | 2024
[5] Bolt-Loosening Monitoring Framework Using an Image-Based Deep Learning and Graphical Model | https://pmc.ncbi.nlm.nih.gov/articles/PMC7349298/ | peer-reviewed primary paper | 2020
[6] A novel anti-loosening bolt looseness diagnosis of bolt connections using a vision-based technique | https://pmc.ncbi.nlm.nih.gov/articles/PMC11106340/ | peer-reviewed primary paper | 2024
[7] DYKEM Cross Check Torque Seal product documentation | https://www.itwprobrands.com/product/cross-check | manufacturer primary documentation | current
[8] Fast-SCNN: Fast Semantic Segmentation Network | https://arxiv.org/abs/1902.04502 | original model paper | 2019
[9] Searching for MobileNetV3 | https://arxiv.org/abs/1905.02244 | original model paper | 2019
[10] ONNX Runtime: Deploy on mobile | https://onnxruntime.ai/docs/tutorials/mobile/ | official deployment documentation | current

## Findings (facts only)

- The strongest anti-loosening-line paper found uses YOLOv10-S for ROI detection, then Fast-SCNN for part and marker segmentation, ellipse/line fitting and spatial projection to estimate the physical relative angle. Its segmentation training used 167 bolt images; it reports 1.145° average angle error, with different error for round and hexagonal nuts. [1]
- The 2022 marked-joint method likewise treats loosening as geometric measurement: Keypoint R-CNN extracts five keypoints, image processing finds the mark ellipse/points, and geometric imaging estimates relative angle under arbitrary viewpoints. [2]
- A train-bolt patent uses raw red-region rectangles and 15° line-angle difference as one embodiment. It also treats a single detected line as normal, which is unsafe for this project because missing/occluded segments can also produce one line. [3]
- A steel-structure study reports a 2.8° loosening discrimination threshold after perspective correction and task-specific contour training. That number belongs to its test setup and is not a universal mechanical limit. [4]
- A reference-image monitoring method compares current and reference bolt angles and sets the upper control limit at three standard deviations of healthy variation. This establishes that a useful alert threshold can be estimated from repeated healthy captures even before loose examples exist. [5]
- Perspective is material: one anti-loosening-bolt study reports uncorrected error increasing above 20° viewing angle and up to 6.5° at a 45° horizontal view; homography correction reduced the maximum average error in its experiment to 1.1°. [6]
- Manufacturer documentation describes torque-seal/witness marks as visual indicators of movement, loosening or tampering. It does not establish remaining preload or a universal angle threshold. [7]
- Fast-SCNN and MobileNetV3 LR-ASPP are explicitly designed for mobile/embedded semantic segmentation. MobileNetV3-LR-ASPP was reported 30% faster than the prior MobileNetV2 R-ASPP at similar Cityscapes accuracy. [8][9]
- ONNX Runtime recommends quantization to reduce model size and states that CPU/XNNPACK/NNAPI performance is model- and device-dependent; the current P20 Pro experiment independently confirmed that NNAPI/XNNPACK can be slower than CPU. [10]

## Local findings

- The current real review pool has 19 independent checkpoints: 7 visually likely aligned, 2 possible displaced, 8 recapture, 2 not a valid bridge. The two possible displaced cases both ended as `INSUFFICIENT`; there are zero controlled real `DISPLACED` truths.
- The only approved state-labelled set is 24 synthetic crops: eight each at approximately 0°, 8° and 24°. These are generated training aids, not threshold-validation evidence.
- The test fixture currently uses `maximum_angle_degrees=8.0` with a strict `>` comparison. On the constructed synthetic metadata this marks only 2/8 nominal 8° samples abnormal because floating-point construction places the other six at or just below 8°. The fixture therefore must not be promoted as a real threshold.
- The current mobile ONNX model only proposes fastener ROIs. No state segmentation/keypoint ONNX model exists.

## Analysis

- A classification-score threshold cannot be calibrated without both classes. A geometry alert threshold can be initialized from measurement repeatability on known-tight captures, but it still cannot establish loose-state recall or precision.
- The decision variable should be the perspective-corrected angular separation of the two mark intersections at the physical joint boundary, not raw image-line angle. Raw line angle changes with viewpoint, paint width and curved surfaces.
- Use a dual threshold rather than one binary cut: a low threshold maximizes recall by routing possible movement to a person; a high threshold prioritizes obvious displacement. Published evidence supports 3° as a defensible provisional low boundary and 15° as a defensible provisional high boundary, but neither is production-calibrated for the user's vehicle.
- If reference size is joint diameter and the two intersections lie on the same circle, chord displacement divided by diameter is `sin(theta/2)`: 3° maps to 0.0262, 8° to 0.0698 and 15° to 0.1305. This is a cross-check, not a substitute for topology-specific projection.
- The safest mobile architecture is a hybrid: lightweight full-image proposal model, one small ROI multi-head segmentation/keypoint model, deterministic geometry and an optional historical-reference matcher. A direct normal/loose classifier would overfit the synthetic appearance and hide failure causes.

## Support snippets / paraphrases for top claims

- Claim: segmentation plus geometric projection is the relevant public baseline. Source [1] separates bolt/nut and marker regions, fits the joint ellipse and two lines, then calculates angle from the intersections.
- Claim: viewpoint correction is mandatory. Source [6] reports materially larger errors without homography at oblique views.
- Claim: healthy-only data can set an alert control limit. Source [5] defines the UCL from three standard deviations of healthy rotation variation.
- Claim: 15° is an engineering precedent, not a universal truth. Source [3] calls 15° one implementation setting and says field thresholds must be chosen per requirement.

## Conflicts / unresolved issues

- Published thresholds range from 2.8° to 15° because the target fastener, reference method, view correction and safety objective differ.
- The user's actual fasteners include planar bolt/nut joints and curved fittings/clamps; one solver cannot safely cover every topology.
- The relation between visible rotation and remaining preload is fastener- and joint-dependent. Image evidence alone should report relative movement, not torque or mechanical safety.

## Gaps

- No controlled before/after real pairs with known angle or maintenance confirmation.
- No repeated tight-state captures to estimate phone/camera/annotator measurement noise.
- No independent physical-point test split across vehicle, lighting, distance and viewpoint.
- No trained state segmentation/keypoint model or measured mobile latency for that model.
