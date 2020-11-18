import os
from jobmonitor.api import (
    kubernetes_schedule_job,
    register_job,
    upload_code_package,
)
from jobmonitor.connections import mongo

excluded_files = [
    "core",
    "output.tmp",
    ".vscode",
    "node_modules",
    "scripts",
    ".git",
    "*.pyc",
    "._*",
    "__pycache__",
    "*.pdf",
    "*.js",
    "*.yaml",
    ".pylintrc",
    ".gitignore",
    ".AppleDouble",
    ".jobignore",
]

code_package, files_uploaded = upload_code_package(".", excludes=excluded_files)
print("Uploaded {} files.".format(len(files_uploaded)))

for noise_level in ["0.00", "0.25", "0.50", "1.00"]:
    cfg = dict(
        pretrained_noise_level=noise_level,
        model_path=f"/raw/vogels/locuslab-smoothing-pretrained-models/imagenet/resnet50/noise_{noise_level}/checkpoint.pth.tar",
    )
    job_id = register_job(
        user="vogels",
        project="adversarial-transfer-learning",
        experiment="does-random-noise-also-help",
        job=f"noise_{noise_level}",
        priority=10,
        config_overrides=cfg,
        runtime_environment={"clone": {"code_package": code_package}, "script": "main.py"},
        annotations={"description": "Using pretrained ImageNet models from https://github.com/locuslab/smoothing, we want to see if Gaussian input perturbations have the same effect as adversarial ones."},
    )

    print(f"jobrun {job_id}")