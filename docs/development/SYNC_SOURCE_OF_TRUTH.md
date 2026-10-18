# Source of truth

The public GitHub `main` branch is the authoritative source of truth for Model Lab 1.3.0. Delivery archives and working-session snapshots are historical inputs, not authoritative repository state.

Generated virtual environments, caches, datasets, build outputs, and nested vendor Git metadata are excluded from the public repository. Native llama.cpp remains a separately bootstrapped and pinned dependency.

When documentation, tests, or release notes disagree with the code on `main`, treat `main` as authoritative and update the stale document rather than reconstructing source state from an older archive.
