"""Manually positioned engineering diagrams for the four A1 modules."""
from pathlib import Path
import sys

sys.path.insert(0, "/home/PuMengYu/.codex/skills/research-figure-style/scripts")
from figure_style import FigureStyle

OUT = Path(__file__).resolve().parent / "figures"


def canvas(style, width=1500, height=720):
    import matplotlib.pyplot as plt
    dpi = style.cfg["output"]["dpi"]
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    return fig, ax


def draw(stem, title, specs, edges, width=1500, height=720):
    style = FigureStyle()
    colors = style.cfg["colors"]
    fig, ax = canvas(style, width, height)
    style.text(ax, width / 2, 42, title, role="heading")
    nodes = {}
    for key, x, y, label, category in specs:
        nodes[key] = style.box(ax, x, y, label, colors[category])
    for source, target, route in edges:
        a, b = nodes[source], nodes[target]
        if isinstance(route, dict):
            if "via_y" in route:
                side = route["side"]
                sy = a["top"] if side == "top" else a["bottom"]
                ty = b["top"] if side == "top" else b["bottom"]
                style.arrow(ax, [(a["x"], sy), (a["x"], route["via_y"]), (b["x"], route["via_y"]), (b["x"], ty)])
            elif "entry_fraction" in route:
                ty = b["top"] + b["h"] * route["entry_fraction"]
                style.arrow(ax, [(a["right"], a["y"]), (route["via_x"], a["y"]), (route["via_x"], ty), (b["left"], ty)])
            else:
                style.arrow(ax, route["points"])
        elif route == "right":
            style.arrow(ax, [(a["right"], a["y"]), (b["left"], b["y"])])
        elif route == "down":
            style.arrow(ax, [(a["x"], a["bottom"]), (b["x"], b["top"])])
        elif route == "up":
            style.arrow(ax, [(a["x"], a["top"]), (b["x"], b["bottom"])])
        else:
            # Explicit orthogonal routing: horizontal, then vertical, then horizontal.
            midx, endy = route
            style.arrow(ax, [(a["right"], a["y"]), (midx, a["y"]), (midx, endy), (b["left"], endy), (b["left"], b["y"])])
    # Explicit identity/bypass connections required by the module forward passes.
    if stem == "06_Bottleneck":
        a, b = nodes["x"], nodes["out"]
        style.arrow(ax, [(a["x"], a["top"]), (a["x"], 115), (b["x"], 115), (b["x"], b["top"])])
    elif stem == "07_Decoder3":
        a, b = nodes["x"], nodes["add"]
        style.arrow(ax, [(a["x"], a["top"]), (a["x"], 115), (b["x"], 115), (b["x"], b["top"])])
    elif stem == "08_Decoder2":
        a, b = nodes["proj"], nodes["mix"]
        style.arrow(ax, [(a["x"], a["top"]), (a["x"], 115), (b["x"], 115), (b["x"], b["top"])])
    result = style.save(fig, OUT / stem)
    print(f"{stem}: audit={result['passed']}")


def detailed_canvas(style, height, width=1000):
    import matplotlib.pyplot as plt
    dpi = style.cfg["output"]["dpi"]
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    return fig, ax


def save_detailed(stem, title, height, specs, edges, width=1000, special_edges=()):
    style = FigureStyle()
    fig, ax = detailed_canvas(style, height, width)
    colors = style.cfg["colors"]
    style.text(ax, width / 2, 42, title, role="heading")
    nodes = {}
    for key, x, y, label, category in specs:
        nodes[key] = style.box(ax, x, y, label, colors[category])
    for source, target in edges:
        a, b = nodes[source], nodes[target]
        if a["x"] == b["x"]:
            style.arrow(ax, [(a["x"], a["bottom"]), (b["x"], b["top"])])
        elif a["y"] < b["y"]:
            mid_y = (a["bottom"] + b["top"]) / 2
            style.arrow(ax, [(a["x"], a["bottom"]), (a["x"], mid_y), (b["x"], mid_y), (b["x"], b["top"])])
        else:
            mid_y = (a["top"] + b["bottom"]) / 2
            style.arrow(ax, [(a["x"], a["top"]), (a["x"], mid_y), (b["x"], mid_y), (b["x"], b["bottom"])])
    for points in special_edges:
        style.arrow(ax, points(nodes))
    result = style.save(fig, OUT / stem)
    print(f"{stem}: audit={result['passed']}")


def main():
    draw("06_Bottleneck", "Bottleneck | Parallel spatial and spectral-channel paths", [
        ("x", 110, 350, "Input x\n512 × 8 × 8", "input"),
        ("pre", 320, 350, "Preprocess\nDWConv → PWConv\n512 → 256", "projection"),
        ("sp", 600, 235, "Spatial path\n128 channels\n3 × Mamba residual", "state"),
        ("spec", 600, 465, "Spectral-channel path\n128 channels\nFFT energy → channel gate", "state"),
        ("fuse", 875, 350, "Branch weighting\nConcat 128 + 128\nPWConv 256 → 256", "projection"),
        ("ffn", 1090, 350, "FFN\n256 → 512 → 256\n(no inner residual)", "state"),
        ("out", 1335, 350, "GELU + output PWConv\n256 → 512\nOuter Add with x", "output"),
    ], [("x", "pre", "right"), ("pre", "sp", (465, 235)), ("pre", "spec", (465, 465)),
        ("sp", "fuse", (760, 235)), ("spec", "fuse", (760, 465)), ("fuse", "ffn", "right"), ("ffn", "out", "right")], 1500, 700)

    draw("07_Decoder3", "Decoder 3 | Directional Wave residual block", [
        ("x", 110, 350, "Input x\n512 × 8 × 8", "input"),
        ("proj", 335, 350, "Depthwise Conv\nLinear 512 → 1536\nSplit u0, v0, z", "projection"),
        ("wave", 620, 350, "Directional Wave\nFFT(u0, v0)\nDamping + wave response", "state"),
        ("gate", 900, 350, "Inverse FFT\nLayerNorm × SiLU(z)\nLinear 512 → 512", "projection"),
        ("scale", 1155, 350, "Scale by gamma\ngamma: 512 channels\ninitialized to 0.001", "state"),
        ("add", 1390, 350, "Residual Add\nx + gamma * Wave(x)\n512 × 8 × 8", "output"),
    ], [("x", "proj", "right"), ("proj", "wave", "right"), ("wave", "gate", "right"), ("gate", "scale", "right"), ("scale", "add", "right")], 1550, 700)
    # Residual bypass from x to the final Add, routed above the processing path.
    # It is drawn after the boxes using an additional, unobstructed orthogonal path.
    # The bypass is represented in the block label and clarified in the Markdown.

    draw("08_Decoder2", "Decoder 2 | DualStage with two weighted corrections", [
        ("cat", 95, 350, "Concat input\nUp2 256 + f2 256\n512 × 16 × 16", "input"),
        ("proj", 285, 350, "Projection U\n1 × 1 Conv\n512 → 256", "projection"),
        ("m", 490, 220, "Mamba correction\nLayerNorm → Mamba\n256 channels", "state"),
        ("r", 490, 480, "RepDW7 correction\nLayerNorm → RepDW7\nGELU → 1 × 1 Conv", "state"),
        ("mix", 835, 350, "Weighted residual sum\nU + c_m M(U) + c_r R(U)\nc_m, c_r initially 0.01", "output"),
        ("out", 1100, 350, "Output\n256 × 16 × 16", "output"),
    ], [("cat", "proj", "right"), ("proj", "m", (385, 220)), ("proj", "r", (385, 480)),
        ("m", "mix", (660, 220)), ("r", "mix", (660, 480)), ("mix", "out", "right")], 1250, 700)

    draw("09_Skip1", "Skip1 | Two inputs, reciprocal channel gates, separate outputs", [
        ("s", 170, 230, "Encoder input s = f0\n64 × 64 × 64", "input"),
        ("d", 170, 510, "Decoder input d = Up0(x1)\n64 × 64 × 64", "input"),
        ("ps", 490, 230, "Side preprocess\n+ Generator G_s", "projection"),
        ("pd", 490, 510, "Side preprocess\n+ Generator G_d", "projection"),
        ("gs", 850, 230, "Cross gate for s\nmeanHW(G_d) → gate_s\nG_s * gate_s", "state"),
        ("gd", 850, 510, "Cross gate for d\nmeanHW(G_s) → gate_d\nG_d * gate_d", "state"),
        ("fs", 1180, 230, "s' Side finish\nResiduals + gamma_s", "projection"),
        ("fd", 1180, 510, "d' Side finish\nResiduals + gamma_d", "projection"),
        ("cat", 1510, 370, "Concat C\n[d', s']\n64 + 64 = 128", "output"),
        ("d0", 1740, 370, "To D0", "output"),
    ], [("s", "ps", "right"), ("d", "pd", "right"), ("ps", "gs", "right"), ("pd", "gd", "right"),
        ("gs", "fs", "right"), ("gd", "fd", "right"),
        ("ps", "gd", {"via_y": 700, "side": "bottom"}), ("pd", "gs", {"via_y": 100, "side": "top"}),
        ("fs", "cat", {"entry_fraction": 0.75, "via_x": 1400}),
        ("fd", "cat", {"entry_fraction": 0.25, "via_x": 1350}), ("cat", "d0", "right")], 1900, 800)

    save_detailed("06a_Bottleneck_preprocess_and_paths", "Bottleneck | Preprocess, serial spatial path, spectral-channel path", 2650, [
        ("x", 500, 125, "Input x\n[B,512,8,8]", "input"),
        ("dw", 500, 265, "DWConv2d\nk=3, p=1, groups=512\nbias=False, zero padding", "projection"),
        ("pw", 500, 405, "PWConv2d 1×1\n512 → 256, bias=True", "projection"),
        ("ln", 500, 545, "LayerNorm(256)\nBCHW → BHWC → BCHW", "state"),
        ("gelu", 500, 675, "GELU", "state"),
        ("u", 500, 805, "u [B,256,8,8]", "input"),
        ("split", 500, 935, "Split_C: 128 + 128", "projection"),
        ("us", 250, 1070, "us [B,128,8,8]", "input"),
        ("uc", 800, 1070, "uc [B,128,8,8]", "input"),
        ("flat", 250, 1200, "Flatten + transpose\n[B,128,8,8] → [B,64,128]", "projection"),
        ("lns", 250, 1330, "LayerNorm(128)", "state"),
        ("mamba", 250, 1460, "Mamba\nd_model=128, d_state=16\nd_conv=4, expand=2, dt_rank=auto", "state"),
        ("adds", 250, 1590, "Add token identity\nt + Mamba(LN(t))", "output"),
        ("restore", 250, 1720, "Transpose + reshape\n[B,64,128] → [B,128,8,8]", "projection"),
        ("fs", 250, 1850, "fs: OneWayResidual output\nRepeat ×3 sequentially", "output"),
        ("fft", 800, 1200, "FFT2 (autocast off)\nfloat32, norm=ortho", "projection"),
        ("power", 800, 1325, "Power\nreal² + imag²", "state"),
        ("mask", 800, 1450, "Apply frequency masks\nLow: |fy|≤0.25 AND |fx|≤0.25\nHigh: complementary bins", "state"),
        ("sum", 800, 1575, "Sum over H,W\neL and eH per channel", "projection"),
        ("stats", 800, 1700, "Stack statistics [B,128,3]\nlog1p(eL), log1p(eH),\neH/(eL+eH+1e-8)", "projection"),
        ("lin1", 800, 1825, "Linear 3 → 16", "state"),
        ("gelus", 800, 1940, "GELU", "state"),
        ("lin2", 800, 2055, "Linear 16 → 1", "state"),
        ("sigmoid", 800, 2170, "Sigmoid", "state"),
        ("scale", 800, 2285, "Multiply by 2\nreshape a → [B,128,1,1]", "projection"),
        ("mul", 800, 2400, "uc × a\nMultiply; broadcast over H,W", "output"),
        ("fc", 800, 2515, "fc [B,128,8,8]", "output"),
    ], [("x", "dw"), ("dw", "pw"), ("pw", "ln"), ("ln", "gelu"), ("gelu", "u"), ("u", "split"),
        ("split", "us"), ("split", "uc"), ("us", "flat"), ("flat", "lns"), ("lns", "mamba"), ("mamba", "adds"),
        ("adds", "restore"), ("restore", "fs"), ("uc", "fft"), ("fft", "power"), ("power", "mask"),
        ("mask", "sum"), ("sum", "stats"), ("stats", "lin1"), ("lin1", "gelus"), ("gelus", "lin2"),
        ("lin2", "sigmoid"), ("sigmoid", "scale"), ("scale", "mul"), ("mul", "fc")],
        width=1200, special_edges=[lambda n: [(n["flat"]["right"], n["flat"]["y"]),
            (420, n["flat"]["y"]), (420, n["adds"]["y"]), (n["adds"]["right"], n["adds"]["y"])],
            lambda n: [(n["uc"]["right"], n["uc"]["y"]), (1120, n["uc"]["y"]),
            (1120, n["mul"]["y"]), (n["mul"]["right"], n["mul"]["y"])]] )

    save_detailed("06b_Bottleneck_branch_weighting_and_output", "Bottleneck | Dynamic branch weighting, fusion, and outer residual", 2780, [
        ("fs", 250, 125, "fs [B,128,8,8]", "input"),
        ("fc", 750, 125, "fc [B,128,8,8]", "input"),
        ("gap_s", 250, 270, "Mean over H,W\n[B,128]", "projection"),
        ("gap_c", 750, 270, "Mean over H,W\n[B,128]", "projection"),
        ("desc", 500, 405, "Concat feature descriptors\n[B,128] + [B,128] → [B,256]", "projection"),
        ("sel1", 500, 535, "Linear 256 → 32", "state"),
        ("selgelu", 500, 655, "GELU", "state"),
        ("sel2", 500, 775, "Linear 32 → 256", "state"),
        ("reshape", 500, 895, "Reshape [B,2,128,1,1]", "projection"),
        ("softmax", 500, 1015, "Softmax over branch axis\nws,wc [B,128,1,1] each", "state"),
        ("mul_s", 250, 1195, "fs × (2ws)\nMultiply; broadcast H,W", "output"),
        ("mul_c", 750, 1195, "fc × (2wc)\nMultiply; broadcast H,W", "output"),
        ("cat", 500, 1325, "Concat_C([2ws·fs, 2wc·fc])\n128 + 128 = 256", "projection"),
        ("fuse", 500, 1450, "PWConv2d 1×1\n256 → 256", "projection"),
        ("norm", 500, 1570, "LayerNorm(256)\nBCHW → BHWC → BCHW", "state"),
        ("ffn1", 500, 1690, "FFN: PWConv2d 1×1\n256 → 512", "projection"),
        ("ffngelu", 500, 1800, "GELU", "state"),
        ("ffn2", 500, 1910, "PWConv2d 1×1\n512 → 256", "projection"),
        ("act", 500, 2020, "GELU", "state"),
        ("out", 500, 2130, "Output PWConv2d 1×1\n256 → 512", "projection"),
        ("gamma", 500, 2245, "Multiply by gamma\n[1,512,1,1]", "state"),
        ("x", 225, 2380, "Identity x\n[B,512,8,8]", "input"),
        ("add", 500, 2380, "Outer residual Add\ny = x + gamma ⊙ out", "output"),
        ("y", 775, 2380, "Output y\n[B,512,8,8]", "output"),
    ], [("fs", "gap_s"), ("fc", "gap_c"), ("gap_s", "desc"), ("gap_c", "desc"),
        ("desc", "sel1"), ("sel1", "selgelu"), ("selgelu", "sel2"), ("sel2", "reshape"),
        ("reshape", "softmax"), ("softmax", "mul_s"), ("softmax", "mul_c"),
        ("mul_s", "cat"), ("mul_c", "cat"), ("cat", "fuse"), ("fuse", "norm"),
        ("norm", "ffn1"), ("ffn1", "ffngelu"), ("ffngelu", "ffn2"), ("ffn2", "act"),
        ("act", "out"), ("out", "gamma"), ("gamma", "add"), ("add", "y")],
        width=1000, special_edges=[
            lambda n: [(n["fs"]["left"], n["fs"]["y"]), (80, n["fs"]["y"]), (80, n["mul_s"]["y"]), (n["mul_s"]["left"], n["mul_s"]["y"])],
            lambda n: [(n["fc"]["right"], n["fc"]["y"]), (920, n["fc"]["y"]), (920, n["mul_c"]["y"]), (n["mul_c"]["right"], n["mul_c"]["y"])],
            lambda n: [(n["x"]["right"], n["x"]["y"]), (n["add"]["left"], n["add"]["y"])],
        ])


if __name__ == "__main__":
    main()
