# Security Policy

## Scope

ML-Framework is a local-first model-data and training framework. Security issues involving credential handling, command execution, filesystem boundaries, network access, dataset ingestion, artifact integrity, dependency supply chain, or localhost control-plane access are in scope.

## Reporting

Please do not publish sensitive details in a public issue when the report could expose credentials, private data, or an exploitable execution path. Contact the repository owner privately through GitHub with reproduction details and the affected commit.

Never include API keys, access tokens, passwords, private dataset contents, or other secrets in a report.

## Design expectations

- Secrets belong in environment variables or local credential storage, never in committed configuration.
- Generated datasets and model artifacts should remain outside source control.
- Network-facing crawlers should preserve configured timeouts, retries, politeness controls, and allow/block lists.
- Local command-center endpoints must preserve localhost binding and request-size/path validation boundaries.
- Untrusted archive/document parsing must remain bounded and fail closed on invalid input.
- Artifact manifests and SHA-256 checks are part of the trust boundary and should not be bypassed silently.
- Dependency changes should be reviewed through the security workflow and dependency audit before release.

## Release-security boundary

A passing source-level security scan does not certify a production dataset, deployment environment, or external-source licensing decision. Those require evidence from the actual release environment and approved data sources.
