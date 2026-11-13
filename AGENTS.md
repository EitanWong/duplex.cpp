# Project conventions

- Daily development uses `dev`. Stable versions move from `dev` to `main` by PR.
- Use squash merges for PRs. Keep additional experimental branches local.
- Preserve the Apple M5 profile and optimized inference paths when adding CUDA support.
- Keep the English and Chinese root READMEs concise and equivalent.
- Never commit model weights, recordings, credentials, local configs or generated builds.
- Do not alter upstream authorship, copyright or license notices.
- Changes to inference need target-hardware evidence for performance and speech quality.
