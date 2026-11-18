"""Prepare review sheets and a manuscript-ready account of the local overlap audit."""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import csv
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data_audit/20261022_image_overlap_verified'
DATASETS=['HAM10000','ISIC2017','ISIC2018_Task1']

def read(name):
    return list(csv.DictReader((OUT/name).open()))

def write(name,rows,fields=None):
    if fields is None:fields=list(rows[0]) if rows else ['no_rows']
    with (OUT/name).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)

def run():
    summary=json.loads((OUT/'audit_summary.json').read_text())
    rows=read('image_inventory.csv');exact=read('exact_overlap_pairs.csv');near=read('perceptual_review_pairs.csv')
    byindex={int(r['index']):r for r in rows}
    matrix=[]
    for da,db in combinations(DATASETS,2):
        a=[r for r in rows if r['dataset']==da];b=[r for r in rows if r['dataset']==db]
        pairs=[r for r in exact if {r['dataset_a'],r['dataset_b']}=={da,db}]
        pp=[r for r in near if {r['dataset_a'],r['dataset_b']}=={da,db}]
        matrix.append({'dataset_a':da,'images_a':len(a),'dataset_b':db,'images_b':len(b),
            'same_image_id':len({r['image_id_a'] for r in pairs if 'same_id' in r['evidence']}),
            'same_file_bytes_pairs':sum('same_file_bytes' in r['evidence'] for r in pairs),
            'same_rgb_pixels_pairs':sum('same_rgb_pixels' in r['evidence'] for r in pairs),
            'renamed_exact_pairs':sum(r['different_ids']=='True' and ('same_file_bytes' in r['evidence'] or 'same_rgb_pixels' in r['evidence']) for r in pairs),
            'perceptual_candidates':len(pp),
            'high_or_geometric_candidates':sum(r['review_category']!='perceptual_only_review' for r in pp)})
    write('dataset_overlap_matrix.csv',matrix)
    splits=Counter()
    for r in exact:
        if r['dataset_a']!=r['dataset_b']:
            splits[(r['dataset_a'],r['official_split_a'],r['dataset_b'],r['official_split_b'])]+=1
    write('official_split_overlap.csv',[dict(dataset_a=k[0],split_a=k[1],dataset_b=k[2],split_b=k[3],overlapping_pairs=v) for k,v in sorted(splits.items())])
    within=[r for r in exact if r['dataset_a']==r['dataset_b']]
    crosssplit=[r for r in within if r['current_split_a']!=r['current_split_b'] and 'not_in_current_custom_split' not in (r['current_split_a'],r['current_split_b'])]
    write('within_dataset_exact_pairs.csv',within)
    write('current_split_exact_overlap.csv',crosssplit)
    provenance=[]
    for p in [ROOT/'data/ham10000_group42/split.csv',ROOT/'data/isic_custom_manifests_seed42/isic2017_split.csv',ROOT/'data/isic_custom_manifests_seed42/isic2018_split.csv']:
        provenance.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT/'split_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    h=[r for r in rows if r['dataset']=='HAM10000']
    groups={s:{r['lesion_id'] for r in h if r['current_split']==s} for s in ('train','val','test')}
    assert all(not groups[a]&groups[b] for a,b in combinations(groups,2))
    # Prioritize HAM candidates and any automated geometric/very-high-similarity candidates.
    priority=[r for r in near if r['review_category']!='perceptual_only_review' or
              ('HAM10000' in (r['dataset_a'],r['dataset_b']) and int(r['distance'])<=4 and
               (r['dataset_a']!=r['dataset_b'] or
                (r['current_split_a']!=r['current_split_b'] and r['lesion_id_a']!=r['lesion_id_b'])))]
    priority.sort(key=lambda r: ('HAM10000' not in (r['dataset_a'],r['dataset_b']),r['review_category']=='perceptual_only_review',-float(r['ssim'])))
    write('priority_visual_review.csv',priority)
    thumbs=np.load(OUT/'thumbnails128.npy',mmap_mode='r')
    reviewdir=OUT/'review_sheets';reviewdir.mkdir(exist_ok=True)
    # All prioritized pairs are exported; image titles preserve the pair indices.
    for start in range(0,len(priority),8):
        if (reviewdir/f'review_{start//8:03d}.jpg').exists():continue
        subset=priority[start:start+8]
        fig,axes=plt.subplots(len(subset),2,figsize=(10,2.8*len(subset)),squeeze=False,constrained_layout=True)
        for j,r in enumerate(subset):
            for col,k in enumerate(('i','j')):
                idx=int(r[k]);item=byindex[idx]
                axes[j,col].imshow(thumbs[idx]);axes[j,col].axis('off')
                axes[j,col].set_title(f"pair {start+j}: {idx} {item['dataset']} {item['image_id']}\n{item['current_split']} | pHash={r['distance']} SSIM={float(r['ssim']):.3f} inliers={r['homography_inliers']}",fontsize=10)
        fig.savefig(reviewdir/f'review_{start//8:03d}.jpg',dpi=100);plt.close(fig)
    summary['dataset_matrix']=matrix
    summary['within_dataset_exact_pairs']=len(within)
    summary['current_split_exact_pairs']=len(crosssplit)
    summary['ham_lesions']={s:len(v) for s,v in groups.items()}
    summary['priority_review_pairs']=len(priority)
    (OUT/'report_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    confirmed=json.loads((OUT/'confirmed_ham_cross_split_variants.json').read_text())
    follow=json.loads((OUT/'original_image_sift_followup.json').read_text())
    summary['targeted_visual_review']={'confirmed_ham_cross_split_variant_pairs':len(confirmed),
        'weak_ham_risk_pairs_original_image_sift_followup':follow['examined'],
        'additional_strong_sift_candidates':len(follow['strong_candidates']),
        'remaining_automated_candidates':'Not all visual candidates manually adjudicated; not evidence of full patient/lesion independence'}
    (OUT/'report_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# HAM10000 / ISIC2017 / ISIC2018 图像重叠审计','',
        '审计日期：2026-10-22。核查原始JPEG输入，不改动图像、标签、划分、模型或既有结果。','',
        '**关键发现：HAM10000当前划分虽然lesion_id无交集，仍存在2对跨划分的同源图像变体：训练—验证1对、训练—测试1对。原图目视与512网格SIFT几何核验一致，不能再将该划分表述为“图像内容完全独立”。昨晚两个模型的训练和评估产物保留，但正式无泄漏比较前需要修订划分并统一重跑。**','',
        '## HAM10000 跨划分变体重复','',
        '| 训练图像 | 对应验证/测试图像 | 元数据病灶ID是否相同 | SIFT内点/匹配 |','|---|---|---|---:|']
    for x in confirmed:
        lines.append(f"| {x['image_id_a']}（{x['split_a']}） | {x['image_id_b']}（{x['split_b']}） | 不同 | {x['inliers']}/{x['ratio_matches']} |")
    lines += ['', '两对均表现为相同摄影内容的颜色/尺度或位置变体，病灶纹理及周围毛发、皮肤纹理或设备文字等细节能对应。这里确认的是同源影像内容，不将目视结果冒充经临床核实的病灶身份。','',
        '![训练与验证之间的变体](HAM_cross_split_focus_0.png)','',
        '![训练与测试之间的变体](HAM_cross_split_focus_1.png)','',
        '[512网格SIFT原始核验](confirmed_ham_cross_split_variants.json) · [第一对特征连线](HAM_cross_split_SIFT_920_5768.png) · [第二对特征连线](HAM_cross_split_SIFT_2721_4730.png)','',
        '## 范围与结果','',
        '共检查16,459条图像记录：HAM10000全部10,015张；ISIC2017官方训练/验证/测试共2,750张；ISIC2018 Task1官方训练/验证/测试共3,694张。此前仅检查2000/2594张训练池的ID；本次扩展为全部本地官方输入。','',
        '| 数据集对 | 相同ID数 | 相同文件字节对 | 相同RGB像素对 | 不同ID的精确重复对 | 感知候选对 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in matrix:
        lines.append(f"| {r['dataset_a']} / {r['dataset_b']} | {r['same_image_id']} | {r['same_file_bytes_pairs']} | {r['same_rgb_pixels_pairs']} | {r['renamed_exact_pairs']} | {r['perceptual_candidates']} |")
    lines += ['', '相同ID表示同一图像记录被复用；相同文件SHA256证明字节副本，RGB像素SHA256证明相同尺寸与解码像素。后三项按图像对统计，不能直接相加当作独立图像数。感知候选不等于已确认重复。','',
        f"同一数据集内部精确重叠共{len(within)}对；其中横跨当前自建train/val/test的为{len(crosssplit)}对，见[具体名单](current_split_exact_overlap.csv)。HAM病灶ID分组为train {len(groups['train'])}、val {len(groups['val'])}、test {len(groups['test'])}个，组间交集为0。",'',
        '## 检查怎么做','',
        '1. 枚举8个原始图像ZIP中的JPEG，排除掩膜、superpixel图与系统文件；保留来源压缩包、成员路径、官方划分及当前自建划分。','',
        '2. 每个JPEG计算文件字节SHA256；解码为RGB后，将颜色模式、宽高和像素字节合并计算第二个SHA256。逐图ID和两种哈希均做全体匹配，覆盖数据集内与数据集间。','',
        '3. 感知筛查使用原图经EXIF方向归一化后的128×128 RGB缓存，再生成256×256灰度参考图，对全图及中心保留90%、75%、50%的区域各构造4个直角旋转与对应镜像，共32个视图。每视图缩放32×32，二维正交DCT取8×8低频，非DC系数中位数二值化（DC位固定0），形成64位存储的pHash。全体记录的32视图分别与全体全图参考视图比较。pHash汉明距离≤10且32×32灰度代理图的零均值归一化相关系数NCC≥0.90，才纳入后续非精确候选。精确哈希始终来自原始JPEG及完整解码RGB，不来自缩略图。','',
        '4. 为避免重复解码大型原图，候选在已按EXIF方向生成的128×128 RGB缩略图上，按pHash最佳视图执行中心裁剪和旋转/镜像，计算灰度SSIM（data_range=255）、RGB RMSE。SSIM≥0.70或pHash距离≤4的候选进一步检查ORB局部特征；ORB最多500特征、edgeThreshold10、patchSize21、Hamming最近邻比值0.75；单应RANSAC阈值3像素、随机种子42。≥20内点、内点比例≥0.6、双图内点凸包覆盖各≥10%时标记几何候选；SSIM≥0.98且RGB RMSE≤12时标记高相似候选。以上均为审计筛查阈值，不用于模型训练或选择分割模型。128网格的候选距离不是原图精确像素匹配证据。','',
        '5. 原始HAM元数据用于病灶组交集检查。本地ISIC2017元数据仅含图像ID、年龄和性别，ISIC2018输入包未附可通用关联HAM的病灶/患者ID。不能据图像相似推断为同患者或同病灶。','',
        f"自动筛查得到{len(near):,}对非精确候选；其中{len(priority)}对导出优先审阅图。完成了上述2对HAM跨划分高置信候选的原图目视确认与512网格SIFT核验。其余{follow['examined']}对低pHash HAM风险候选追加原图512网格SIFT检查，没有新增达到预设强几何阈值的图像对。该追加检查条件为SIFT最多4000特征、L2比值0.70、RANSAC3像素/seed42、至少12内点、内点比例至少0.8、双图内点覆盖各至少10%、内点中位误差不超过2像素。",'',
        '其他自动高相似/几何候选没有全部逐对目视定性，尤其不能把ISIC内部所有自动候选都计成已确认的变体重复。全部原始指标见[候选表](perceptual_review_pairs.csv)，审阅图见[review_sheets](review_sheets/)，[576对原图SIFT追加检查](original_image_sift_followup.json)。','',
        '## 对实验的影响','',
        'HAM与ISIC之间的结果应分别区分“独立训练评估”和“把一个数据集训练出的权重转移到另一个数据集”。对于ISIC2017与2018，如使用前者训练权重初始化后者，或混合两者训练，必须对目标测试集做去重审计；同一ID跨数据集复用不能称为独立外部样本。','',
        '本次不自动删除样本或重划已有测试集。当前已确认HAM训练—验证与训练—测试之间的变体重复，应把已确认重复边与原lesion_id关系合并成连通组，再形成统一固定的修订划分；预算按最终训练集重新核算，并统一重跑受影响的模型。不能仅在论文中补一句“已去重”而沿用原数字。本次未实施修订或重跑。','',
        '## 论文可用表述与限制','',
        '可先使用的方法描述：为核查数据集重叠，我们对全部本地图像进行标识符、文件SHA256及解码RGB像素SHA256匹配，并采用包含直角旋转、镜像及多尺度中心裁剪的感知哈希筛查，进一步使用结构相似性和局部特征几何一致性检查候选。HAM10000内部划分以lesion_id为分组单位。具体重叠数量及人工核查结果在审计结果表中分别报告。','',
        '论文不能写成“已经证明三个数据集完全独立”或“无任何同病灶重叠”。本审计不穷尽任意位置裁剪、强形变、重新拍摄、明显光照变化等情形；缺少共享患者/病灶标识时，无法严格排除同病灶不同照片。','',
        '## 可复核文件','',
        '- [逐图来源与SHA256](image_inventory.csv)','- [精确重叠图像对](exact_overlap_pairs.csv)',
        '- [官方划分间重叠](official_split_overlap.csv)','- [当前划分SHA256](split_provenance.json)',
        '- [自动核查配置和汇总](audit_summary.json)','- [报告统计](report_summary.json)',
        '- [筛查脚本](../../audit_image_overlap.py)','- [报告与审阅图脚本](../../report_image_overlap.py)','',
        '数据来源：[HAM10000官方数据](https://doi.org/10.7910/DVN/DBW86T)、[ISIC官方挑战数据](https://challenge.isic-archive.com/data/)。HAM分割掩膜来自作者补充文件`HAM10000_segmentations_lesion_tschandl.zip`，本地元数据记录发布于2020-06-22，使用时需引用Tschandl等的[Nature Medicine 2020论文](https://doi.org/10.1038/s41591-020-0942-0)。此审计比较输入图像，不以掩膜相似作为独立影像证据。','']
    (OUT/'数据重叠审计.md').write_text('\n'.join(lines))
    paper=['# 论文方法与结果文字草稿','',
        '**使用状态：审计方法和发现可引用；尚未实施去重后的修订划分与重跑，不能写成已经完成数据清理。**','',
        '## 中文方法','',
        '我们对HAM10000的10015张图像、ISIC2017官方训练/验证/测试共2750张图像及ISIC2018 Task1官方训练/验证/测试共3694张图像开展重叠审计。首先匹配图像标识符、原文件SHA256及包含尺寸信息的解码RGB像素SHA256，以发现同名或改名的精确副本。进一步采用旋转、镜像和多尺度中心裁剪视图的感知哈希与归一化相关性筛查，并以SSIM、ORB及单应几何一致性对候选分级。对HAM跨划分高置信候选进行原图目视及512×512网格SIFT复核；对576对额外低pHash风险候选开展同规格SIFT检查。完整阈值、版本、图像对和原始来源记录随审计材料提供。','',
        '## 中文结果','',
        'HAM10000与本地ISIC2017/2018 Task1完整图像池之间均未发现同ID或文件/像素精确副本，所用感知与几何筛查亦未检出跨这些数据集的高置信变体候选，但这不构成患者或病灶层面完全独立的证明。ISIC2017与ISIC2018 Task1存在2450个同ID且内容完全相同的复用记录，另有5对不同ID的精确副本。HAM10000内部存在2对不同ID的精确副本，均在同一训练子集中且各自同lesion_id。然而，在lesion_id不重叠的既有HAM划分中，我们仍确认2对同源图像变体分别跨越训练—验证与训练—测试，说明仅按现有病灶标识分组不足以完全排除影像内容重叠。ISIC2017内部另发现6对不同ID精确副本，其中4对跨官方训练/验证/测试边界，2对均在测试集中。','',
        '## English methods draft','',
        'We audited all locally available dermoscopic inputs: 10,015 HAM10000 images, 2,750 ISIC 2017 images, and 3,694 ISIC 2018 Task 1 images, including the official training, validation, and test pools. Exact overlap was assessed using image identifiers, SHA-256 hashes of the original files, and dimension-aware SHA-256 hashes of decoded RGB pixels. Approximate screening used perceptual hashes over rotations, reflections, and centered crops, together with normalized cross-correlation. Candidates were further characterized using structural similarity and local-feature homography consistency. High-confidence HAM10000 cross-split candidates underwent original-image visual review and SIFT verification on a 512-by-512 grid. An additional 576 low-pHash HAM risk candidates underwent the same SIFT follow-up. Detailed thresholds, source manifests, split hashes, and pair-level results are retained in the audit package.','',
        '## English results draft','',
        'No matching image identifiers or byte-/pixel-exact copies were detected between HAM10000 and the complete local ISIC 2017 or ISIC 2018 Task 1 pools. The specified approximate screening also identified no high-confidence cross-dataset variant candidates involving HAM10000; however, these findings do not establish complete patient- or lesion-level independence. ISIC 2017 and ISIC 2018 Task 1 shared 2,450 same-identifier, byte-identical images, with five additional differently named exact-copy pairs. Within HAM10000, two differently named exact-copy pairs occurred within the training subset and shared their respective lesion identifiers. Importantly, visual and geometric checks identified two additional same-source photographic variant pairs with different lesion identifiers, crossing the training–validation and training–test boundaries, respectively. Thus, disjoint recorded lesion identifiers alone did not ensure image-content independence in the existing split.','',
        '## 正式论文提交前还要完成','',
        '固定重复组的处理规则及修订划分；更新划分哈希与训练预算；对所有比较模型统一重跑。之后才能补写“如何去重、去重后样本数和最终性能”，不得把本草稿的审计发现改写成已完成的清理措施。','',
        '[完整数据审计](数据重叠审计.md)','']
    (OUT/'论文_数据独立性方法与结果草稿.md').write_text('\n'.join(paper))
    (OUT/'progress.json').write_text(json.dumps({'stage':'automated_audit_complete_with_targeted_manual_review','images':len(rows),
        'confirmed_ham_cross_split_variant_pairs':2,'additional_original_image_sift_checks':follow['examined'],
        'all_candidates_manually_adjudicated':False,'split_repair_and_retraining_performed':False},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'matrix':matrix,'within_exact':len(within),'crosssplit_exact':len(crosssplit),'priority_visual_pairs':len(priority)},ensure_ascii=False))

if __name__=='__main__':run()
