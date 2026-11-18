from model.convneur import ConvNeur

def build_model(scale="m1"):
    if scale == "m1":
        model = ConvNeur(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], mem_dim=80, chunk_size=196)
    elif scale == "m2":
        model = ConvNeur(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], mem_dim=96, chunk_size=196)
    elif scale == "m3":
        model = ConvNeur(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], mem_dim=128, chunk_size=196)
    elif scale == "m4":
        model = ConvNeur(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], mem_dim=160, chunk_size=196)
    else:
        raise ValueError(f"Scale {scale} not supported.")
    return model