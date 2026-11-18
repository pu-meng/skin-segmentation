from pathlib import Path
import subprocess,json,sys
r=Path('/home/PuMengYu/experiments/skin_segmentation');out=r/'log/Conv_Attention_Mamba_整网对比/多block频谱对比'
py='/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python'
for kind in ['conv','transformer','mamba']:
 (out/'status.json').write_text(json.dumps({'status':'extracting','current_model':kind})+'\n')
 subprocess.run([py,'-B','-u',str(r/'source/comparison_mixers/analyze_blocks.py'),'--kind',kind,'--device','cuda:1','--output',str(out/kind)],check=True)
subprocess.run([py,'-B','-u',str(r/'source/comparison_mixers/report_blocks.py')],check=True)
