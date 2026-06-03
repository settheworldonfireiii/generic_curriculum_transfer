from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from gct.config.schema import BackendConfig


@dataclass(frozen=True)
class BackendSubmission:
    backend: str
    command: list[str]
    job_id: str | None = None


class LocalBackend:
    def submit(self, command: list[str], cwd: Path | None = None) -> BackendSubmission:
        proc = subprocess.Popen(command, cwd=cwd)
        return BackendSubmission("local", command, str(proc.pid))


class SlurmBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    def submit(self, command: list[str], cwd: Path | None = None) -> BackendSubmission:
        wrapped = " ".join(_shell_quote(part) for part in command)
        slurm_command = [
            "sbatch",
            "--parsable",
            f"--partition={self.config.slurm_partition}",
            f"--gres={self.config.slurm_gres}",
            f"--time={self.config.slurm_time}",
            "--wrap",
            wrapped,
        ]
        proc = subprocess.run(slurm_command, cwd=cwd, check=True, capture_output=True, text=True)
        return BackendSubmission("slurm", slurm_command, proc.stdout.strip())


class IbmCloudBackend:
    """Command builder for IBM Cloud deployment.

    GPU workloads still run through the same `gct` CLI after SSH/cloud-init.
    This backend intentionally emits commands instead of hiding cloud state.
    """

    def create_vsi_command(self, name: str, profile: str, image: str, ssh_key_id: str, subnet_id: str) -> list[str]:
        return [
            "ibmcloud",
            "is",
            "instance-create",
            name,
            "vpc-id",
            "zone-name",
            profile,
            subnet_id,
            "--image",
            image,
            "--keys",
            ssh_key_id,
        ]


def _shell_quote(value: str) -> str:
    if value.replace("_", "").replace("-", "").replace("/", "").replace(".", "").isalnum():
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"

