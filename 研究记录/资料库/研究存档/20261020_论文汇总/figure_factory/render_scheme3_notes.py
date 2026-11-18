"""将方案3独立说明编译成离线HTML，内嵌SVG，无网络字体或公式依赖。"""
from pathlib import Path
from html import escape, unescape
import base64
import re
import markdown

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / '方案3_问题定义与思路检索/论文问题定义与思路检索.md'
OUT = DOC.with_name('方案3_图文与公式排版.html')

def render(doc=DOC, out=OUT):
    source = doc.read_text(encoding='utf-8')
    updated = re.search(r'^\s*更新[：:]\s*(\d{4}-\d{2}-\d{2})', source, re.M)
    stamp_date = updated.group(1) if updated else '2026-10-13'
    body = markdown.markdown(source, extensions=['tables', 'fenced_code', 'toc'], output_format='html')
    body = re.sub(r'(<p(?:\s[^>]*)?>)\u3000{2}', r'\1', body)
    def embed(match):
        path = (doc.parent / unescape(match[1])).resolve()
        if path.suffix != '.svg' or not path.is_file():
            raise ValueError(f'Missing SVG: {path}')
        uri = 'data:image/svg+xml;base64,' + base64.b64encode(path.read_bytes()).decode()
        return f'src="{uri}"'
    body = re.sub(r'src="([^"]+)"', embed, body)
    def card(match):
        lines = match[1].strip().splitlines()
        return '<div class="calculation">' + ''.join('<div>' + line + '</div>' for line in lines) + '</div>'
    body = re.sub(r'<pre><code class="language-text">(.*?)</code></pre>', card, body, flags=re.S)
    # 使用浏览器原生MathML渲染主式；不依赖外部JS或TeX插件。
    formula = '''<aside class="formula"><div>阶段3·修正模块：输出头之后、分类之前加入外部修正</div>
    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
    <msub><mi>z</mi><mtext>修正后</mtext></msub><mo>=</mo><mi>z</mi><mo>+</mo><mi>S</mi><mo>×</mo><msub><mi>C</mi><mtext>base</mtext></msub>
    </math><p>输出头三类分数 ＋ 统一强度 × 对齐输出网格的三类修正图</p></aside>'''
    body = body.replace('<a id="question"></a>', formula + '<a id="question"></a>')
    css = '''
    *{box-sizing:border-box}body{margin:0;background:#edf1f5;color:#233448;font:17px/1.8 "Noto Sans CJK SC","Microsoft YaHei",sans-serif}
    main{max-width:1080px;margin:28px auto;padding:44px 54px;background:white;border-radius:12px}
    h1{font-size:30px}h2{font-size:25px;margin-top:48px;border-bottom:2px solid #d8e3ef;padding-bottom:10px}h3{font-size:21px;margin-top:32px}
    p{margin:16px 0}a{color:#225a95}strong{font-weight:700}img{max-width:100%;height:auto;display:block;margin:22px auto;border:1px solid #dce3ea}
    table{border-collapse:collapse;width:100%;margin:22px 0;font-size:16px}th,td{border:1px solid #d9e2ec;padding:12px 14px;text-align:left;vertical-align:top}th{background:#e9f1f9}tr:nth-child(even){background:#f7f9fc}
    code{font:0.92em/1.7 "DejaVu Sans Mono",monospace;background:#f1f4f8;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}pre{background:#f1f4f8;padding:20px;overflow:auto}pre code{padding:0}
    .calculation{background:#f1f5fc;border-left:4px solid #5078ad;padding:18px 24px;margin:20px 0;overflow:auto;font:16px/2 "DejaVu Sans Mono","Noto Sans CJK SC",monospace;white-space:pre}
    .formula{background:#eaf5ef;border:1px solid #b7d7c2;padding:24px;text-align:center;margin:28px 0}.formula math{font-size:30px;margin:14px}.formula p{margin-bottom:0;font-size:16px}
    /* chinese-paragraph-indent */
main > p{text-indent:2em}
main > p:has(img),main > p:has(> small){text-indent:0}
.stamp{color:#53687d;font-size:15px;padding:12px 18px;background:#f3f6fa}
    @media(max-width:700px){main{margin:0;padding:22px 18px}table{font-size:14px}th,td{padding:8px}.calculation{font-size:14px;padding:12px}}
    @media print{body{background:white}main{max-width:none;margin:0;padding:0}h2,h3{break-after:avoid}img,table,.formula{break-inside:avoid}}
    '''
    result = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + escape(source.splitlines()[0].removeprefix('# ')) + '</title><style>' + css + '</style></head><body><main><div class="stamp">' + stamp_date + ' · 当前候选设计，尚无真实CT完整实现和效果证据 · SVG与公式均可离线显示</div>' + body + '</main></body></html>'
    out.write_text(result, encoding='utf-8')
    print(out)

if __name__ == '__main__':
    render()
    for filename, html_name in [('方法1_局部纠错记忆与有效性过滤.md', '方法1_图文与公式排版.html'), ('方法2_类别平均特征检索与注入.md', '方法2_图文与公式排版.html')]:
        render(DOC.with_name(filename), DOC.with_name(html_name))
