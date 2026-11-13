# Development

`main` is the stable integration branch and the default GitHub branch. Daily work
happens on `dev`. Push completed changes to `dev`; promote a tested version through
a **`dev` → `main` pull request**, using **Squash and merge**.

Keep experimental branches local and merge their useful changes into `dev`.
After a release PR is squashed, synchronize `dev` with `main` before continuing:

```bash
git switch dev
git fetch origin
git merge origin/main
git push origin dev
```

Do not rewrite a published `dev` branch. Avoid additional remote branches.
Tag releases from `main` after validation on the target hardware.

For inference changes, report hardware, model hashes and precision, context sizes,
sampling settings, and both cold and warm measurements. Compare audio continuity,
interruption behavior, peak memory and speech quality alongside latency. Preserve
the Apple baseline when adding NVIDIA paths; keep hardware-specific defaults
explicit. Do not report upstream benchmark numbers as results of this project.

Run `python3 scripts/check-repository.py` and relevant runtime tests before pushing.
The repository CI performs structural checks and model-free tests; live inference
and target-hardware acceptance are separate checks. Review upstream contribution
policies before proposing changes to their repositories.

## 开发约定

`main` 为稳定版本与默认分支，日常开发统一在 `dev` 进行。完成开发后推送
`dev`，验证稳定后提交 `dev` → `main` PR，统一 Squash and merge。
实验分支只保留在本地，有效改动合入 `dev`；不重写已发布的 `dev` 历史。
每次 squash 合并后，按上面的命令将 `main` 合回 `dev`。
性能改动需同时记录延迟、内存、连续性、打断行为与语音质量，并保留 Apple 基线。
