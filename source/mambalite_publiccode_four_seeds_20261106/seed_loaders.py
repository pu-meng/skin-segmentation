from ham_fast_pipeline import *
def make_fast_loaders(config, api, workers=2, validation_batch=8, seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = api.MambaLiteUNet()
        loader_state = torch.get_rng_state()
        del reference
    generator = torch.Generator()
    generator.set_state(loader_state)
    data = Path(config['data']['split_file']).parent
    loaders = {}
    for split in ('train', 'val'):
        dataset = PlannedDataset(data, split)
        sampler = AugmentationPlanSampler(api.ReferenceRandomSampler(dataset, generator)) if split == 'train' else None
        kwargs = dict(batch_size=8 if split == 'train' else validation_batch,
                      sampler=sampler, num_workers=workers, pin_memory=True,
                      drop_last=False, generator=torch.Generator().manual_seed(seed*1000+1),
                      worker_init_fn=worker_init)
        if workers:
            kwargs.update(persistent_workers=True, prefetch_factor=2,
                          multiprocessing_context='spawn')
        loaders[split] = EpochLoader(DataLoader(dataset, **kwargs), generator)
    if len(loaders['train']) != config['training']['updates_per_epoch']:
        raise ValueError('Training update budget changed')
    kwargs = dict(batch_size=1, num_workers=workers, pin_memory=True,
                  generator=torch.Generator().manual_seed(seed*1000+2), worker_init_fn=worker_init)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context='spawn')
    # Independent iterator seeds avoid consuming the training sampler's stream.
    precise = DataLoader(loaders['val'].dataset, **kwargs)
    loaders['val_precise'] = precise
    return loaders, generator
