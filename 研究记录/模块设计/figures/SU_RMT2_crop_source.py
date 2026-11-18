from pathlib import Path
import fitz,json,hashlib
root=Path(__file__).parent
pdf=Path('C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/attachments/SU-RMT-2.pdf')
doc=fitz.open(pdf)
out=root/'original_crops';out.mkdir(exist_ok=True)
items=[('SU_RMT2_original_Fig2_p4',4,[36,52,563,518]),('SU_RMT2_original_Fig3_p4',4,[36,522,563,658]),('SU_RMT2_original_Fig4_p5',5,[36,52,563,218]),('SU_RMT2_original_Fig5_p6',6,[305,52,563,247])]
rows=[]
for name,page,rect in items:
 p=out/(name+'.png');pix=doc[page-1].get_pixmap(matrix=fitz.Matrix(4,4),clip=fitz.Rect(rect),alpha=False);pix.save(p)
 rows.append(dict(file=p.name,page=page,crop_pdf_points=rect,width=pix.width,height=pix.height,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
manifest=dict(source='SU-RMT-2.pdf',source_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),crops=rows)
(out/'SU_RMT2_original_crops.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
