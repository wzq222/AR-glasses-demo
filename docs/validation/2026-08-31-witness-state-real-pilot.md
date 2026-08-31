# 真实防松线松动状态首批复核与安全基线

## 结论

本轮工作对象已经从“检查点检测”切换到“防松线所指示的相对位移状态”。代码新增安全状态融合基线，
固定输出`ALIGNED / DISPLACED / DAMAGED_MARK / INSUFFICIENT`，并保留
`POSSIBLE_DISPLACED`人工复核提示。单一二维几何异常不得直接输出`DISPLACED`；只有几何与独立的
局部状态模型或历史基准共同支持时才可升级。生产阈值未用真实受控样本标定时，任何样本都只能输出
`INSUFFICIENT`。

当前通用几何基线只开放`bolt_head_plate / nut_plate`两个平面拓扑。`nut_stud / double_nut /
fitting_pipe / clamp_pipe`需要圆柱投影、双零件绑定等专用求解器；未实现前即使首审肉眼接近对齐，程序
也固定返回`TOPOLOGY_SOLVER_UNAVAILABLE -> INSUFFICIENT`，不会套用平面角度公式。

Git外首批真实状态复核包覆盖19个独立真实检查点、3个批次，全部提供原像素1×、nearest 2×和4×证据。
Codex完成首审后得到：

| 首审提示 | 数量 | 含义 |
|---|---:|---|
| `LIKELY_ALIGNED` | 7 | 跨moving/fixed两侧的标记可见，首审未见明确相对位移 |
| `POSSIBLE_DISPLACED` | 2 | 两侧标记方向差明显，必须进入隐藏结论端点二审 |
| `RECAPTURE` | 8 | 像素、清晰度、遮挡或多目标归属不足 |
| `NOT_A_BRIDGE` | 2 | 单边涂点或大面积结构漆，不能用于松动判断 |

这19张不是受控松动实验，故首审没有把任何记录写成正式`ALIGNED/DISPLACED`真值。当前结果证明状态
复核能够筛出疑似位移和拒判样本，不证明松动识别准确率。

## 隐藏结论二审

首审的`ref-09/ref-13`已进入独立二审包。二审任务只包含哈希校验后的原像素1×/2×/4×视图和坐标网格，
不包含首审提示或理由。结果为：

| 检查点 | 二审结果 | 证据 |
|---|---|---|
| `ref-09` | `INSUFFICIENT` | 红漆跨过螺栓头和固定板，但两侧均为宽涂层，没有唯一、可重复的线轴和端点 |
| `ref-13` | `INSUFFICIENT` | 人工端点重算夹角约1.394°、间隙比0，但原图目标太小，端点置信度仅0.55，低于0.60门 |

因此首审的2个`POSSIBLE_DISPLACED`均未升级为`DISPLACED`，现有19个真实参考点中仍有0个可用
`DISPLACED`真值。二审审计器要求结果绑定二审pack、task和正式真值哈希；单张历史观察即使人工写成
`ALIGNED/DISPLACED`也会被`DECIDABLE_STATE_REQUIRES_CONTROLLED_PAIR`拒绝。审计时还会重新核验
原图1×/2×/4×证据和坐标网格SHA-256，pack发布后任一证据被改写都会拒绝。当前审计为2张已复核、
1张端点完整、0张训练可用，状态分布为`INSUFFICIENT=2`，训练门保持关闭。

## 发现并修正的错误路径

最初复用ImageGen像素选择器时，程序对19张真实裁剪中的18张都拟合出了“两段线”。原图叠加复核表明，
大量拟合来自锈迹、铜色表面、橙色底漆和相邻结构线；例如`ref-09`、`ref-14`、`ref-18`的选中区域
远大于真实防松线。根因是该选择器原本依赖生成前后的像素差，不能在没有baseline的真实照片上把红黄
油漆与锈蚀/铜色材料分开。

因此废弃`review-packs/witness-state-v1/real-reference-pilot`的18条几何建议；v1.1完成语义修正，最终
以事务式发布的v1.2作为当前复核包：

- 自动几何建议固定为0，不把颜色掩膜提升为端点；
- 颜色掩膜只保留为`trusted_for_geometry=false`的弱视觉提示；
- 固定侧/活动侧、两段端点和状态必须由原图首审/二审绑定；
- 无端点、未知拓扑、单边标记或未标定阈值均为`INSUFFICIENT`。
- manifest与每张裁剪只读取一次并绑定SHA-256；全部资产在同目录staging中完成，正式真值复核通过后
  才原子发布，任一失败不会在正式路径留下半包。

## 可复现资产

- 安全复核包：
  `E:/crrc_vision_data/review-packs/witness-state-v1.2/real-reference-pilot/manifest.json`
- Codex首审：
  `E:/crrc_vision_data/review-packs/witness-state-v1.2/real-reference-pilot/codex-first-pass.json`
- 首审SHA-256：
  `D3636E5655B1ACE5667B01E81DE7F9B799DC08EE379B8F59C68B6AB54210BDC6`
- 隐藏二审pack：
  `E:/crrc_vision_data/review-packs/witness-state-v1.2/second-pass-ref09-ref13/manifest.json`
- 二审pack SHA-256：
  `3F1F8BE1217BAB46AEC59CA1C284E0E5E7DDD50186994EE63AB2EE0052100338`
- 二审结论SHA-256：
  `7DB859555A09AABBC84CDED015AB987D80B403CBA8B1BD99C1073465A7A7A084`
- 二审审计SHA-256：
  `BFC98417CE60BBD0E172132D990AFD9B0184B36B3284A181B0532D79F2B70C9F`
- 状态融合：`ml/src/crrc_vision/witness_state.py`
- 真实状态复核包：`ml/src/crrc_vision/witness_state_review_pack.py`
- 二审结果审计：`ml/src/crrc_vision/witness_state_second_review.py`
- 正式真值SHA-256：
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`
- `python -m pytest ml/tests -q`：337 passed、1 skipped。
- `gradlew testDebugUnitTest`与`gradlew assembleDebug`：通过。
- Debug APK：28,275,171 bytes，SHA-256
  `D745B6682F3E7A445C0AA3AA896DB5512E7C2DAD2429609891A0CF00F6F6DC6D`。

## 下一门

1. 现场选同一物理检查点，在维护人员控制下采集紧固验收态、已知小角位移和恢复态；固定相机距离、
   曝光和多视角，并记录`point_id`、实际转角/操作、维护人员确认和两段端点。
2. 真实成对样本到位后标定几何阈值，训练局部paint/moving/fixed分割与端点模型；ImageGen只做外观增强。
3. 独立真实测试必须覆盖`DISPLACED`召回、误把`DISPLACED`判为`ALIGNED`的关键错误率，以及所有拒判。
