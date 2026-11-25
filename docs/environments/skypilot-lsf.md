# SkyPilot on LSF — deprecated

SkyPilot's LSF cloud (`default_cloud: lsf`) is deprecated, and these docs no longer cover it. The
SkyPilot release gbserver pins has no LSF cloud, so `default_cloud: lsf` needs a SkyPilot build
that adds one. Move LSF builds to **SkyPilot on SLURM**. See [the native LSF page](lsf.md)
for what to change, and [SkyPilot on SLURM](skypilot-slurm.md) for environment setup.
