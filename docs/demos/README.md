# Demos

End-to-end demos that exercise Granite.Build against real workloads. Each one
brings up a self-contained environment, submits one or more builds, and tears
down cleanly when you're done.

> If you just want to see a build run on your laptop, the
> [getting-started guide](../getting-started.md) and
> [`samples/standalone/standalone-quickstart/`](../../samples/standalone/standalone-quickstart/)
> are smaller and faster.

- [Standalone Docker demo](docker-demo.md) — TRL fine-tuning and unitxt evaluation in Docker containers via the standalone server. No cluster or cloud credentials needed.
- [SLURM demo (via SkyPilot)](skypilot-slurm-demo.md) — the same workload on a local Docker SLURM cluster via SkyPilot, with artifact push to MinIO. Runs entirely locally.
- [Granite 4.0 Nano — SFT + Eval on AWS](granite4_nano.md) — fine-tune Granite 4.0 350M on AWS and run the full evaluation suite via SkyPilot.
