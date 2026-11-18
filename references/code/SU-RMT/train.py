import os

from Models import get_model
from source.config import get_args
from source.tools import describe_model
from source.train_isic import train_isic
from source.train_synapse import train_synapse
from source.train_polyp import train_polyp
from source.log import setup_isic_logger, setup_synapse_logger, setup_polyp_logger
from source.utils.random_seed import setup_seed


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    setup_seed(args.seed)
    classes, channels = get_dataset_params(args.dataset_name)

    original_id = args.id
    id_suffixes = []

    if args.loss_strategy != "Normal":
        id_suffixes.append(args.loss_strategy)

    if getattr(args, 'use_ema', False):
        id_suffixes.append("EMA")

    if id_suffixes:
        args.id = f"{args.id}-{'-'.join(id_suffixes)}"

    model_class = get_model(original_id)
    model = model_class(classes=classes, channels=channels).cuda()

    if args.dataset_name in ['ISIC2018', 'ISIC2017']:
        log_setup = setup_isic_logger(args)
        validation_results_csv = log_setup['validation_results_csv']
        test_results_csv = log_setup['test_results_csv']
        logger = log_setup['logger']

        describe_model(model=model, model_name=original_id, out_size=args.out_size, channels=channels)

        train_isic(
            args=args,
            validation_results_csv=validation_results_csv,
            test_results_csv=test_results_csv,
            model=model,
            logger=logger,
        )
    elif args.dataset_name == 'Synapse':
        log_paths = setup_synapse_logger(args)

        describe_model(model=model, model_name=original_id, out_size=args.out_size, channels=channels)

        train_synapse(
            args=args,
            **log_paths,
            model=model,
        )

    elif args.dataset_name == 'Polyp':
        log_paths = setup_polyp_logger(args)

        describe_model(model=model, model_name=original_id, out_size=args.out_size, channels=channels)

        train_polyp(
            args=args,
            **log_paths,
            model=model,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_name}")


def get_dataset_params(dataset_name):
    """
    Return appropriate classes and channels parameters based on dataset name

    Args:
        dataset_name: Dataset name

    Returns:
        (classes, channels): Number of classes and input channels
    """
    dataset_params = {
        'ISIC2018': (2, 3),  # ISIC2018: Background + lesion, RGB input
        'ISIC2017': (2, 3),  # ISIC2017: Background + lesion, RGB input
        'Polyp': (2, 3),     # Polyp: Background + polyp, RGB input
        'Synapse': (9, 1),   # Synapse: Background + 8 organs, grayscale input
    }

    if dataset_name not in dataset_params:
        raise ValueError(f"Unknown dataset: {dataset_name}, cannot determine classes and channels parameters")

    return dataset_params[dataset_name]


if __name__ == "__main__":
    main()
