# 防松标记适用范围核查

## 结论

防松线（更准确地说是torque stripe / witness mark / movement indicator）不只用于普通螺栓和螺母。
正规用途还包括压缩管接头、螺纹接头、管卡以及其他已设定扭矩、张力或位置的组件。Henkel的官方
产品说明直接列出compression fittings、studs、nuts、parts和assemblies；NASA归档工程规范也要求
nuts and fittings拧紧后涂torque stripe。[1][2]

轨道场景同样不能简单等同于“只看螺丝”。铁路相关专利把防松标记用于双螺母之外的腕臂、支持管、
线夹和防抽脱部位；轨道交通操作规程的转载文本还指向管接头及管卡的独立防松标识规程。[3][4]

## 对项目的约束

- 保留螺栓/螺母和有明确螺纹或夹紧运动副的管接头/管卡。
- 只有标记跨越moving/fixed两侧且拓扑可解释，才可判`ALIGNED / DISPLACED / DAMAGED_MARK`。
- 单边涂点、普通色线、热缩管、警示漆和锈迹只能作`INSUFFICIENT / LOOKALIKE`。
- 防松线只说明发生过相对运动或拆动，不能单凭照片证明剩余扭矩或预紧力。

## 局限

不同车型和工位的工艺文件可能收得更窄；客户SOP与点位清单仍是最终范围。现有公开证据能证明
“并非只有螺丝”，但不能替代中车具体车型的作业指导书。

## References

[1] Henkel, LOCTITE SF 7414 product page. <https://www.henkel-adhesives.com/sk/en/product/industrial-inks-and-coatings/loctite_sf_7414.html>

[2] NASA NTRS, Equipment Specification EQ 2-228 A. <https://ntrs.nasa.gov/api/citations/19710001585/downloads/19710001585.pdf>

[3] CN112014400A, high-speed railway contact-network anti-loosening marking method. <https://patents.google.com/patent/CN112014400A/zh>

[4] 轨道交通装备螺栓紧固防松标识操作规程转载页. <https://www.doczj.com/doc/1e1874262.html>
