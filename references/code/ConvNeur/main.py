import argparse
import json

from timm.utils import ModelEma
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

import utils.utils as utils
from utils.utils import *
from utils.datasets import build_dataset
from utils.engine import train_one_epoch, evaluate
from utils.optim_factory import create_optimizer
from model import build_model


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default=False, type=bool,  help='Perform evaluation only')
    parser.add_argument('--batch_size', default=64, type=int, help='Per GPU batch size')
    parser.add_argument('--update_freq', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=20)

    parser.add_argument('--model_scale', default='m1', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float,)
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+')
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    parser.add_argument('--data_path', default='/data/imagenet', type=str)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=10, type=int)

    parser.add_argument('--dist_eval', default=True, type=bool, help='Enabling distributed evaluation')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    return parser


def main(args):
    init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                     mode='batch', label_smoothing=0.1, num_classes=args.nb_classes)

    model = build_model(scale=args.model_scale)
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)["model"]
        model_dict = model.state_dict()
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in checkpoint.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        print(f"No load key from pretrain: {no_load_key}")
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    model.to(device)
    model.to(device)
    model_ema = ModelEma(model, decay=0.9999, device='', resume='')
    print("Using EMA with decay = 0.9999")

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=-1,
    )

    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    criterion = SoftTargetCrossEntropy()
    print("criterion = %s" % str(criterion))

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq
        )
        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            test_stats_ema = evaluate(data_loader_val, model_ema.ema, device)
            print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
            if max_accuracy_ema < test_stats_ema["acc1"]:
                max_accuracy_ema = test_stats_ema["acc1"]
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
            log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
