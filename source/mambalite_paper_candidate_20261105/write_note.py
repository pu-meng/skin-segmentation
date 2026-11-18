from pathlib import Path
import hashlib,json
root=Path('/home/PuMengYu/experiments/skin_segmentation')
p=root/'source/mambalite_paper_candidate_20261105'
text='''# MambaLiteUNet 论文对齐候选：定义待裁决，未开训

CPU 诊断13项通过、0失败；不是完整候选网络验收。GPU1共享与单卡顺序五次已获授权，资源不再按空闲等待；现有E1.3未动。

已核实：仅6个AMF分支残差skip_scale对应α；公开原值1，内存改0不影响其余参数。实际残差操作数是归一化后的x_chunk。

四向SS2D已由现有上游源码核清并CPU参考前后向：行优先、列优先及各自反向；x_proj、dt_proj、A、D分方向独立；反向重排后求和，再LN。不是四个完整1D Mamba取平均。TinyViM的SS2D另含分频/池化，不能整块移植来冒称本论文。

尚须科学负责人明确一个版本：
- 图2(b)是Proj→DWConv3×3→SiLU→SS2D→LN，与SiLU(Proj K)相乘后再Proj；§3.2正文却写SiLU在DWConv前，未明确最后Proj。
- 正文投影矩阵C×C；上游VMamba默认内部2C。采用图示的C宽，还是完整VMamba默认2C，公开MambaLiteUNet没有可对应SS2D版本。
- AMF残差沿用公开的归一化X_k，还是按公式解释为原始X_k，需明确；改α初值不能替代这项定义。

见protocol_draft.json：seed42–46、每次从头300轮、batch8、AdamW .001/wd.01、Tmax50每轮step、验证最低seg loss选best、每seed完整2020测试/CSV/summary/report/12图，五次均值与样本std。采样/增强必须私有seed化，不能使用共享硬编码42的loader。

由于上述关系未唯一确定，未拼装候选整网、未执行GPU batch8、未部署训练控制器/tmux；不能说已排队或已启动。定版后仍须真实CUDA batch8预检通过；OOM失败阻断，不减batch或CPU回退。

证据：cpu_diagnostic_report.json、alpha_original_negative_check.json、upstream_SS2D_class_snapshot.py、resource_snapshot.json、prepare_evidence.py。旧模型/共享训练代码的哈希保持不变；0.494M未被视为调通道目标；旧成绩保留。
'''
(p/'README.md').write_text(text)
entry=root/'log/八模型对比/results.md'
old=entry.read_text()
line='2026-11-05：[MambaLiteUNet论文对齐五次候选准备](../../source/mambalite_paper_candidate_20261105/README.md)：CPU算子诊断通过；论文图文/内部宽度关系待裁决，未开训；旧成绩保留。\n\n'
assert line not in old
new=old.replace('# HAM10000 V5 八模型实验汇总\n\n','# HAM10000 V5 八模型实验汇总\n\n'+line,1)
assert new!=old
entry.write_text(new)
# External write readback verification.
assert (p/'README.md').read_text()==text
assert line in entry.read_text()
print(json.dumps({'README_readback':True,'existing_result_entry_short_line_readback':True,'README':str(p/'README.md'),'entry':str(entry)},ensure_ascii=False))
