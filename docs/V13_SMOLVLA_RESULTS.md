# Official SmolVLA results

Status: **NOT_RUN**. No official SmolVLA weights, text/vision processor or LIBERO dataset were downloaded, loaded, trained or evaluated. The local CNN+ACT and CNN+DDPM models are **not** SmolVLA and contain no language encoder. No VLA success rate is reported.

The current Windows environment lacks the Linux-only [LeRobot LIBERO integration](https://huggingface.co/docs/lerobot/main/en/libero); local PyTorch is CPU-only. The cloud migration scripts prepare a pinned Linux environment, but a paid instance has not been authorized or launched. Before frozen SmolVLA evaluation, verify its official source commit, checkpoint revision and hash, license, training data, image keys, 8D proprioception, 7D action mode, task instructions, hard reset and per-task initial states. Fine-tuning requires a separate actual GPU memory/time probe and budget approval. See `V13_LIBERO_INTEGRATION.md`.
