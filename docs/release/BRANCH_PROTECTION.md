# Main branch protection

The repository's release policy requires `main` to accept production changes only through pull requests with the automated release gates passing.

Required status checks:

- CI
- Security
- CodeQL Advanced

The branch should require branches to be up to date before merge and should not allow force-pushes or branch deletion.

The GitHub repository settings remain the enforcement point. This file records the intended policy so a fresh repository audit can distinguish source-tree policy from live GitHub administration state.
