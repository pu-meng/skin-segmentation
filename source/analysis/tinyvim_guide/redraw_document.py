"""Rebuild the figures displayed by the TinyViM/Mamba research document.
Uses the established generators, including the two unchanged stack declarations.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'source/bottleneck_mamba'))


def main():
    from draw_baseline import draw as baseline
    from draw_block_details import draw_all as blocks
    from draw_v2_projection_variants import draw_all as projections
    from draw_frequency_variants import draw_all as frequencies
    from draw_v4_overview import draw as v4, detail as v4_detail
    from draw_shared_adgbc import draw as shared_adgbc, detail as shared_adgbc_detail
    from draw_adgbc_paper import integration, assignment, aggregate
    from draw_shallow_skip_conv import draw as shallow_skip_conv
    from draw_remaining_controls import draw_all as remaining_controls
    from mamba_figure_layout import draw as mamba
    import draw as guide

    baseline()
    baseline(current=True)
    blocks()
    projections()
    frequencies()
    v4()
    v4_detail()
    shared_adgbc()
    shared_adgbc_detail()
    integration()
    assignment()
    aggregate()
    shallow_skip_conv()
    remaining_controls()
    # Reuse the canonical declarations without rebuilding unrelated old guides.
    tree = ast.parse(Path(guide.__file__).read_text())
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
    wanted = {'32_TinyViM单方向状态计算', '34_最浅SkipLoRA计算'}
    found = set()
    for node in main_node.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id=='stack'
                and node.value.args and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value in wanted):
            found.add(node.value.args[0].value)
            code = compile(ast.Module(body=[node], type_ignores=[]), guide.__file__, 'exec')
            exec(code, {'stack':guide.stack})
    assert found == wanted, found
    for variant, stem in [
        ('original','12_原始Mamba_同版式流程图'),
        ('original','14_原始Mamba_逐块维度与下标'),
        ('v1','13_第一版_卷积生成参数'),
        ('v2','11_第二版_写入项去掉Delta'),
        ('v2','15_第二版_逐块维度与下标'),
        ('v2light','16_第二版轻量候选_共享A与平均读出'),
        ('v3','10_第三版_D乘内容门控'),
    ]:
        mamba(variant,stem=stem)


if __name__ == '__main__':
    main()
