# GitHub upload notes

This directory is intended to be the repository root. It contains no active `.github/workflows` file, so a normal HTTPS push does not need the special PAT workflow permission.

After creating an empty GitHub repository, the minimal upload sequence is:

```bash
git init -b main
git add -A
git commit -m "Initial GitHub release"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Replace `OWNER/REPOSITORY` with the real repository path. If the repository already contains commits, clone it first and copy this package into the clone instead of blindly force-pushing.

To enable GitHub Actions later, read `docs/GITHUB_ACTIONS_OPTIONAL.md`.
