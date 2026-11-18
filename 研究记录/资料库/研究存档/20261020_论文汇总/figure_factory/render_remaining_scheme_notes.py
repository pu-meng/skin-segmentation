"""将方案5两个候选方法的完整Markdown渲染为同一份离线HTML。"""
from pathlib import Path
from html import escape, unescape
import base64
import re
import markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = '''
*{box-sizing:border-box}body{margin:0;background:#edf1f5;color:#233448;font:17px/1.8 "Noto Sans CJK SC","Microsoft YaHei",sans-serif}
main{max-width:1080px;margin:28px auto;padding:44px 54px;background:white;border-radius:12px}
h1{font-size:30px}h2{font-size:25px;margin-top:48px;border-bottom:2px solid #d8e3ef;padding-bottom:10px}h3{font-size:21px;margin-top:32px}
p{margin:16px 0}a{color:#225a95}img{max-width:100%;height:auto;display:block;margin:22px auto;border:1px solid #dce3ea}
table{border-collapse:collapse;width:100%;margin:22px 0;font-size:16px}th,td{border:1px solid #d9e2ec;padding:12px 14px;text-align:left;vertical-align:top}th{background:#e9f1f9}tr:nth-child(even){background:#f7f9fc}
code{font:0.92em/1.7 "DejaVu Sans Mono",monospace;background:#f1f4f8;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}pre{background:#f1f4f8;padding:20px;overflow:auto}
/* chinese-paragraph-indent */
main > p{text-indent:2em}
main > p:has(img),main > p:has(> small){text-indent:0}
.stamp{color:#53687d;font-size:15px;padding:12px 18px;background:#f3f6fa}
@media(max-width:700px){main{margin:0;padding:22px 18px}table{display:block;overflow-x:auto;font-size:14px}th,td{padding:8px}}
@media print{body{background:white}main{max-width:none;margin:0;padding:0}h2,h3{break-after:avoid}img{break-inside:avoid}}
'''


def render(number):
    doc = ROOT / f'方案{number}_问题定义与思路检索/论文问题定义与思路检索.md'
    source = doc.read_text(encoding='utf-8')
    body = markdown.markdown(source, extensions=['tables', 'fenced_code', 'toc'], output_format='html')
    body = re.sub(r'(<p(?:\s[^>]*)?>)\u3000{2}', r'\1', body)

    def embed(match):
        path = (doc.parent / unescape(match[1])).resolve()
        if path.suffix != '.svg' or not path.is_file():
            raise ValueError(f'Missing SVG: {path}')
        uri = 'data:image/svg+xml;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
        return f'src="{uri}"'

    body = re.sub(r'src="([^"]+)"', embed, body)
    title = escape(source.splitlines()[0].removeprefix('# '))
    result = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width,initial-scale=1">'
              f'<title>{title}｜图文排版</title><style>{CSS}</style></head><body><main>'
              '<div class="stamp">2026-10-13 · 候选设计说明 · 图已内嵌，可离线显示 · 算例不代表实测结果</div>'
              + body + '</main></body></html>')
    out = doc.with_name(f'方案{number}_图文与公式排版.html')
    out.write_text(result, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    for scheme_number in (5,):
        render(scheme_number)
