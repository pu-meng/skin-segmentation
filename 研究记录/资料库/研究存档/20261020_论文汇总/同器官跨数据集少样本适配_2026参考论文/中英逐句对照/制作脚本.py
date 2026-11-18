from pathlib import Path
import concurrent.futures, hashlib, html, io, json, re, time, unicodedata
import requests, fitz
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path('/home/PuMengYu/论文汇总/同器官跨数据集少样本适配_2026参考论文')
OUT=ROOT/'中英逐句对照'; OUT.mkdir(exist_ok=True)
CACHE=OUT/'翻译缓存.json'
cache=json.loads(CACHE.read_text()) if CACHE.exists() else {}
params=PunktParameters(); params.abbrev_types={'fig','figs','eq','eqs','sec','secs','tab','tabs','et','al','e.g','i.e','vs','dr','mr','prof','resp','approx'}
tokenizer=PunktSentenceTokenizer(params)
pdfmetrics.registerFont(TTFont('English','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('Chinese','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',subfontIndex=0))
styles={
 'en':ParagraphStyle('en',fontName='English',fontSize=9,leading=13.5,spaceAfter=3,textColor=colors.HexColor('#23364a'),wordWrap='CJK'),
 'zh':ParagraphStyle('zh',fontName='Chinese',fontSize=10,leading=15,spaceAfter=10,textColor=colors.HexColor('#111111'),wordWrap='CJK'),
 'title':ParagraphStyle('title',fontName='Chinese',fontSize=16,leading=23,spaceAfter=16,wordWrap='CJK'),
 'note':ParagraphStyle('note',fontName='Chinese',fontSize=9,leading=14,spaceAfter=10,wordWrap='CJK')}

def clean(t):
 t=unicodedata.normalize('NFKC',t)
 t=re.sub(r'([A-Za-z])[-‐]\s*\n\s*([a-z])',r'\1\2',t)
 return re.sub(r'\s+',' ',t).strip()

def extract(doc):
 pages=[]; refs=False
 for pi,p in enumerate(doc):
  w,h=p.rect.width,p.rect.height
  blocks=[b for b in p.get_text('blocks') if b[6]==0]
  blocks=[b for b in blocks if b[1]>23 and b[3]<h-24]
  if pi==0:
   abs_y=next((b[1] for b in blocks if clean(b[4])=='Abstract'),0)
   # Author names, affiliations and email remain on the original page.
   blocks=[b for b in blocks if b[1]>=abs_y or any(k in b[4] for k in ['FAAR:', 'Improvise, Adapt', 'SD-FSMIS:'])]
  # Column-major reading, with genuinely full-width titles before columns.
  full=[];left=[];right=[]
  for b in blocks:
   if b[0]<w*.25 and b[2]>w*.75 and b[1]<h*.30:full.append(b)
   elif (b[0]+b[2])/2<w*.51:left.append(b)
   else:right.append(b)
  ordered=sorted(full,key=lambda b:b[1])+sorted(left,key=lambda b:b[1])+sorted(right,key=lambda b:b[1])
  sentences=[]
  for b in ordered:
   text=clean(b[4])
   if not text or re.fullmatch(r'[\d\W]+',text):continue
   if any(x in text for x in ['This CVPR paper is the Open Access','This WACV paper is the Open Access','Except for this watermark','the final published version']):continue
   if re.match(r'^References\b',text):refs=True
   if refs:continue
   # Tables/equations remain faithfully available on original vector pages.
   words=re.findall(r'\b[A-Za-z]{2,}\b',text)
   if len(words)<3 and 'Abstract' not in text:continue
   if pi==0 and text.startswith('FAAR:'):
    text='FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection'
   if pi==0 and text.startswith('Improvise, Adapt'):
    text='Improvise, Adapt, Overcome — Telescopic Adapters for Efficient Fine-tuning of Vision Language Models in Medical Imaging'
   if pi==0 and text.startswith('SD-FSMIS:'):
    text='SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation'
   for sent in tokenizer.tokenize(text):
    sent=sent.strip()
    if sent and len(re.findall(r'[A-Za-z]',sent))>=5:sentences.append(sent)
  pages.append(sentences)
 return pages

def fix_zh(t):
 for a,b in [('脊椎骨','主干网络'),('骨干网络','主干网络'),('骨干网','主干网络'),('骨干模型','主干模型'),('少镜头','少样本'),('小镜头','少样本'),('面具','掩膜'),('地面真相','真实标注'),('地面实况','真实标注'),('香草注意力','标准注意力'),('香草适配器','标准适配器'),('香草适配','标准适配'),('排名','秩'),('光谱','频谱'),('性能驱动秩收缩','性能驱动的秩缩减'),('绩效驱动秩缩减','性能驱动的秩缩减'),('任务谱金字塔','任务频谱金字塔'),('减少了多达 9 倍的参数数量','将需要训练的参数量最多降至约九分之一'),('变压器','Transformer'),('Telescope Adapters','伸缩式适配器'),('Telescopic Adapters','伸缩式适配器'),('伸缩适配器','伸缩式适配器')]:t=t.replace(a,b)
 return t

def key(s):return hashlib.sha256(s.encode()).hexdigest()

def translate(batch):
 params=[('client','dict-chrome-ex'),('sl','en'),('tl','zh-CN')]+[('q',x) for x in batch]
 for attempt in range(5):
  try:
   r=requests.get('https://clients5.google.com/translate_a/t',params=params,timeout=45);r.raise_for_status();data=r.json()
   if not isinstance(data,list) or len(data)!=len(batch):raise ValueError('translation alignment mismatch')
   if any(not isinstance(t,str) for t in data):raise ValueError('unexpected translation format')
   return [(s,fix_zh(t)) for s,t in zip(batch,data)]
  except Exception:
   if attempt==4:raise
   time.sleep(1+attempt*2)

all_docs=[];todo=[];seen=set()
for path in sorted(ROOT.glob('*.pdf')):
 doc=fitz.open(path);pages=extract(doc);doc.close()
 all_docs.append((path,pages))
 for page in pages:
  for s in page:
   k=key(s)
   if k not in cache and k not in seen:todo.append(s);seen.add(k)
 print('EXTRACTED',path.name,'pages',len(pages),'sentences',sum(map(len,pages)),flush=True)

batches=[];b=[];size=0
for s in todo:
 # URL encoding plus unicode math can expand requests substantially.
 if b and (size+len(s)>2100 or len(b)>=10):batches.append(b);b=[];size=0
 b.append(s);size+=len(s)
if b:batches.append(b)
print('TRANSLATING',len(todo),'sentences in',len(batches),'batches',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 futures=[pool.submit(translate,b) for b in batches]
 for n,f in enumerate(concurrent.futures.as_completed(futures),1):
  for s,t in f.result():cache[key(s)]={'en':s,'zh':t}
  CACHE.write_text(json.dumps(cache,ensure_ascii=False,indent=2))
  if n%10==0 or n==len(futures):print('TRANSLATED',n,'/',len(futures),flush=True)

manual={
 'Abstract':'摘要',
 'SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation':'SD-FSMIS：将Stable Diffusion适配于少样本医学图像分割',
 'Improvise, Adapt, Overcome — Telescopic Adapters for Efficient Fine-tuning of Vision Language Models in Medical Imaging':'随机应变、适应挑战、克服困难：用于医学影像视觉语言模型高效微调的伸缩式适配器',
 'Furthermore, our model produces high-quality segmentation results even when applied to the more challenging cross-domain FSMIS tasks.':'此外，即使用于更具挑战性的跨域少样本医学图像分割任务，我们的模型仍能生成高质量的分割结果。',
 'In this work, we proposed SD-FSMIS, a novel approach that leverages Stable Diffusion for the FSMIS task.':'本文提出了SD-FSMIS，一种利用Stable Diffusion完成少样本医学图像分割任务的新方法。',
 'FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection':'FAAR：通过自动秩选择实现高效的频率感知多任务微调',
}
for item in cache.values():
 item['zh']=fix_zh(manual.get(item['en'],item['zh']))
 if re.search(r'\branks?\b|low.rank',item['en'],re.I):
  item['zh']=item['zh'].replace('等级','秩').replace('阶数','秩').replace('1 级更新','秩为1的更新')
 if not item['zh']:item['zh']=item['en']
CACHE.write_text(json.dumps(cache,ensure_ascii=False,indent=2))

def footer(c,doc):
 c.setFont('Chinese',8);c.setFillColor(colors.HexColor('#666666'))
 c.drawString(40,24,'中英逐句对照 · 机器翻译初稿 · 公式与表格数值请核对原文页')
 c.setFont('English',8);c.drawRightString(555,24,str(doc.page))

audit=[]
for path,pages in all_docs:
 source=fitz.open(path);merged=fitz.open();toc=[];md=[];sentence_count=0
 title=path.stem
 for pi,sentences in enumerate(pages):
  toc.append([1,f'原文第{pi+1}页',len(merged)+1]);merged.insert_pdf(source,from_page=pi,to_page=pi)
  if not sentences:continue
  stream=io.BytesIO()
  pdf=SimpleDocTemplate(stream,pagesize=(595.28,841.89),rightMargin=40,leftMargin=40,topMargin=42,bottomMargin=45)
  story=[Paragraph(f'{html.escape(title)}<br/>原文第{pi+1}页：逐句对照',styles['title'])]
  if pi==0:
   story.append(Paragraph('阅读方式：每个原文页后附该页的英文—中文逐句对照。原文页保留图表、公式和数值；对照页按双栏阅读顺序重新排版。参考文献保留原文。中文由机器翻译生成并做术语替换，未经全文人工逐句审校；数学符号和复杂表格请以原文页为准。',styles['note']))
  md.append(f'\n## 原文第{pi+1}页\n')
  for s in sentences:
   z=cache[key(s)]['zh'];sentence_count+=1
   pair=[Paragraph(html.escape(s),styles['en']),Paragraph(html.escape(z),styles['zh'])]
   if len(s)<2200:story.append(KeepTogether(pair))
   else:story.extend(pair)
   md.extend([s,'',z,''])
  pdf.build(story,onFirstPage=footer,onLaterPages=footer)
  trans=fitz.open(stream=stream.getvalue(),filetype='pdf');merged.insert_pdf(trans);trans.close()
 merged.set_toc(toc);merged.set_metadata({'title':title+' 中英逐句对照','subject':'English sentence followed by Chinese machine translation; original pages preserved','author':'Bilingual reading copy'})
 dest=OUT/(title+'_中英逐句对照.pdf');merged.save(dest,garbage=4,deflate=True)
 (OUT/(title+'_中英逐句对照.md')).write_text('# '+title+'：中英逐句对照\n\n机器翻译初稿；原文图表公式见同名PDF。参考文献保留原文。\n'+'\n'.join(md))
 missing=[s for pg in pages for s in pg if not cache.get(key(s),{}).get('zh')]
 audit.append({'file':dest.name,'source_pages':len(source),'output_pages':len(merged),'sentence_pairs':sentence_count,'missing_translations':len(missing),'original_pdf_preserved':True,'translation_type':'machine draft with terminology substitutions; not fully human reviewed'})
 print('SAVED',dest.name,'pages',len(merged),'pairs',sentence_count,flush=True)
 merged.close();source.close()
(OUT/'制作核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('COMPLETE',json.dumps(audit,ensure_ascii=False),flush=True)
