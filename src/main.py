import argparse
import hashlib
import os

import cox.store
import numpy as np
import torch as ch
from cox import utils
from robustness import datasets, defaults, model_utils, train
from robustness.tools import helpers
from torch import nn
from torchvision import models

from utils import constants as cs
from utils import fine_tunify, transfer_datasets

config = dict(
    arch="resnet50",
    freeze_level=-1,
    lr=0.001,
    weight_decay=0.0005,
    pretrained_noise_level=0.25,
    batch_size=64,
    epochs=150,
    step_lr=50,
    model_path="/raw/vogels/locuslab-smoothing-pretrained-models/imagenet/resnet50/noise_0.25/checkpoint.pth.tar",
    dataset="cifar10",
    additional_hidden=0,
    adv_eval=None,
    adv_train=0,
    attack_lr=None,
    attack_steps=None,
    cifar10_cifar10=False,
    config_path=None,
    constraint=None,
    custom_eps_multiplier=None,
    custom_lr_multiplier=None,
    data_aug=1,
    eps=None,
    eval_only=0,
    log_iters=5,
    lr_interpolation="step",
    mixed_precision=0,
    momentum=0.9,
    no_replace_last_layer=False,
    no_tqdm=1,
    out_dir="outdir",
    per_class_accuracy=False,
    pytorch_pretrained=False,
    random_restarts=None,
    random_start=None,
    resume=False,
    resume_optimizer=0,
    save_ckpt_iters=-1,
    step_lr_gamma=0.1,
    subset=None,
    use_best=None,
    workers=30,
)


class Bunch(object):
    """Let's you access items in a dict by with dict.item"""
    def __init__(self, adict):
        self.__dict__.update(adict)


output_dir = "./output.tmp"

pytorch_models = {
    "alexnet": models.alexnet,
    "vgg16": models.vgg16,
    "vgg16_bn": models.vgg16_bn,
    "squeezenet": models.squeezenet1_0,
    "densenet": models.densenet161,
    "shufflenet": models.shufflenet_v2_x1_0,
    "mobilenet": models.mobilenet_v2,
    "resnext50_32x4d": models.resnext50_32x4d,
    "mnasnet": models.mnasnet1_0,
}

class StoreWrapper(cox.store.Store):
    """
    Hack to save results in Thijs' framework
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_epoch = -1
        add_scalar_to_tensorboard = self.tensorboard.add_scalar
        def add_scalar(name, value, epoch, *args, **kwargs):
            add_scalar_to_tensorboard(name, value, epoch, *args, **kwargs)

            # report progress
            if epoch > self.last_epoch:
                self.last_epoch = epoch
                log_info({"state.progress": (epoch+1) / config["epochs"]})

            adv_type, split, metric = name.split("_")  # by assumption
            if isinstance(value, ch.Tensor):
                value = value.cpu().item()
            log_metric(metric, {"value": value, "epoch": epoch+1}, {"split": split, "adversarial_type": adv_type})

        self.tensorboard.add_scalar = add_scalar

def main():
    args = Bunch(config)

    print("Translating model file")
    path_hash = hashlib.md5(args.model_path.encode("utf-8")).hexdigest()
    translated_model_path = f"/tmp/checkpoint{path_hash}"
    g = ch.load(args.model_path)
    sd = {}
    for k, v in g["state_dict"].items():
        kk = k[len("1.module."):]
        sd[f"module.attacker.model.{kk}"] = v
        sd[f"module.model.{kk}"] = v
    ch.save({"state_dict": sd, "epoch": g["epoch"]}, translated_model_path)
    args.__dict__["model_path"] = translated_model_path
    print("Done translating")

    # Create store and log the args
    store = StoreWrapper(os.path.join(output_dir, "cox"))
    if "metadata" not in store.keys:
        args_dict = args.__dict__
        schema = cox.store.schema_from_dict(args_dict)
        store.add_table("metadata", schema)
        store["metadata"].append_row(args_dict)
    else:
        print("[Found existing metadata in store. Skipping this part.]")

    ds, train_loader, validation_loader = get_dataset_and_loaders(args)

    if args.per_class_accuracy:
        assert args.dataset in [
            "pets",
            "caltech101",
            "caltech256",
            "flowers",
            "aircraft",
        ], f"Per-class accuracy not supported for the {args.dataset} dataset."

        # VERY IMPORTANT
        # We report the per-class accuracy using the validation
        # set distribution. So ignore the training accuracy (as you will see it go
        # beyond 100. Don't freak out, it doesn't really capture anything),
        # just look at the validation accuarcy
        args.custom_accuracy = get_per_class_accuracy(args, validation_loader)

    model, checkpoint = get_model(args, ds)

    if args.eval_only:
        return train.eval_model(args, model, validation_loader, store=store)

    update_params = freeze_model(model, freeze_level=args.freeze_level)

    log_info({"state.progress": 0.0})
    print(f"Dataset: {args.dataset} | Model: {args.arch}")
    train.train_model(
        args,
        model,
        (train_loader, validation_loader),
        store=store,
        checkpoint=checkpoint,
        update_params=update_params,
    )


def get_per_class_accuracy(args, loader):
    """Returns the custom per_class_accuracy function. When using this custom function         
    look at only the validation accuracy. Ignore trainig set accuracy.
    """

    def _get_class_weights(args, loader):
        """Returns the distribution of classes in a given dataset.
        """
        if args.dataset in ["pets", "flowers"]:
            targets = loader.dataset.targets

        elif args.dataset in ["caltech101", "caltech256"]:
            targets = np.array(
                [loader.dataset.ds.dataset.y[idx] for idx in loader.dataset.ds.indices]
            )

        elif args.dataset == "aircraft":
            targets = [s[1] for s in loader.dataset.samples]

        counts = np.unique(targets, return_counts=True)[1]
        class_weights = counts.sum() / (counts * len(counts))
        return ch.Tensor(class_weights)

    class_weights = _get_class_weights(args, loader)

    def custom_acc(logits, labels):
        """Returns the top1 accuracy, weighted by the class distribution.
        This is important when evaluating an unbalanced dataset. 
        """
        batch_size = labels.size(0)
        maxk = min(5, logits.shape[-1])
        prec1, _ = helpers.accuracy(logits, labels, topk=(1, maxk), exact=True)

        normal_prec1 = prec1.sum(0, keepdim=True).mul_(100 / batch_size)
        weighted_prec1 = prec1 * class_weights[labels.cpu()].cuda()
        weighted_prec1 = weighted_prec1.sum(0, keepdim=True).mul_(100 / batch_size)

        return weighted_prec1.item(), normal_prec1.item()

    return custom_acc


def get_dataset_and_loaders(args):
    """Given arguments, returns a datasets object and the train and validation loaders.
    """
    if args.dataset in ["imagenet", "stylized_imagenet"]:
        ds = datasets.ImageNet(args.data)
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=8
        )
    elif args.cifar10_cifar10:
        ds = datasets.CIFAR(os.path.join(os.getenv("DATA"), "data"))
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=8
        )
    else:
        ds, (train_loader, validation_loader) = transfer_datasets.make_loaders(
            args.dataset, args.batch_size, 8, args.subset
        )
        if type(ds) == int:
            new_ds = datasets.CIFAR(os.path.join(os.getenv("DATA"), "data"))
            new_ds.num_classes = ds
            new_ds.mean = ch.tensor([0.0, 0.0, 0.0])
            new_ds.std = ch.tensor([1.0, 1.0, 1.0])
            ds = new_ds
    return ds, train_loader, validation_loader


def resume_finetuning_from_checkpoint(args, ds, finetuned_model_path):
    """Given arguments, dataset object and a finetuned model_path, returns a model
    with loaded weights and returns the checkpoint necessary for resuming training.
    """
    print("[Resuming finetuning from a checkpoint...]")
    if (
        args.dataset in list(transfer_datasets.DS_TO_FUNC.keys())
        and not args.cifar10_cifar10
    ):
        model, _ = model_utils.make_and_restore_model(
            arch=pytorch_models[args.arch](args.pytorch_pretrained)
            if args.arch in pytorch_models.keys()
            else args.arch,
            dataset=datasets.ImageNet(""),
            add_custom_forward=args.arch in pytorch_models.keys(),
        )
        while hasattr(model, "model"):
            model = model.model
        model = fine_tunify.ft(args.arch, model, ds.num_classes, args.additional_hidden)
        model, checkpoint = model_utils.make_and_restore_model(
            arch=model,
            dataset=ds,
            resume_path=finetuned_model_path,
            add_custom_forward=args.additional_hidden > 0
            or args.arch in pytorch_models.keys(),
        )
    else:
        model, checkpoint = model_utils.make_and_restore_model(
            arch=args.arch, dataset=ds, resume_path=finetuned_model_path
        )
    return model, checkpoint


def get_model(args, ds):
    """Given arguments and a dataset object, returns an ImageNet model (with appropriate last layer changes to 
    fit the target dataset) and a checkpoint.The checkpoint is set to None if noe resuming training.
    """
    finetuned_model_path = os.path.join(
        args.out_dir, "checkpoint.pt.latest"
    )
    if args.resume and os.path.isfile(finetuned_model_path):
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, finetuned_model_path
        )
    else:
        if (
            args.dataset in list(transfer_datasets.DS_TO_FUNC.keys())
            and not args.cifar10_cifar10
        ):
            model, _ = model_utils.make_and_restore_model(
                arch=pytorch_models[args.arch](args.pytorch_pretrained)
                if args.arch in pytorch_models.keys()
                else args.arch,
                dataset=datasets.ImageNet(""),
                resume_path=args.model_path,
                pytorch_pretrained=args.pytorch_pretrained,
                add_custom_forward=args.arch in pytorch_models.keys(),
            )
            checkpoint = None
        else:
            model, _ = model_utils.make_and_restore_model(
                arch=args.arch,
                dataset=ds,
                resume_path=args.model_path,
                pytorch_pretrained=args.pytorch_pretrained,
            )
            checkpoint = None

        if not args.no_replace_last_layer and not args.eval_only:
            print(
                f"[Replacing the last layer with {args.additional_hidden} "
                f"hidden layers and 1 classification layer that fits the {args.dataset} dataset.]"
            )
            while hasattr(model, "model"):
                model = model.model
            model = fine_tunify.ft(
                args.arch, model, ds.num_classes, args.additional_hidden
            )
            model, checkpoint = model_utils.make_and_restore_model(
                arch=model,
                dataset=ds,
                add_custom_forward=args.additional_hidden > 0
                or args.arch in pytorch_models.keys(),
            )
        else:
            print("[NOT replacing the last layer]")
    return model, checkpoint


def freeze_model(model, freeze_level):
    """
    Freezes up to args.freeze_level layers of the model (assumes a resnet model)
    """
    # Freeze layers according to args.freeze-level
    update_params = None
    if freeze_level != -1:
        # assumes a resnet architecture
        assert len(
            [
                name
                for name, _ in list(model.named_parameters())
                if f"layer{freeze_level}" in name
            ]
        ), "unknown freeze level (only {1,2,3,4} for ResNets)"
        update_params = []
        freeze = True
        for name, param in model.named_parameters():
            print(name, param.size())

            if not freeze and f"layer{freeze_level}" not in name:
                print(f"[Appending the params of {name} to the update list]")
                update_params.append(param)
            else:
                param.requires_grad = False

            if freeze and f"layer{freeze_level}" in name:
                # if the freeze level is detected stop freezing onwards
                freeze = False
    return update_params


def log_metric(name, values, tags={}):
    """Log timeseries data
       This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags))


def log_info(info_dict):
    """Add any information to MongoDB
       This function will be overwritten when called through run.py"""
    pass


if __name__ == "__main__":
    main()
