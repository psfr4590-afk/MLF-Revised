# Model Lab Verification Checklist

## Automated
- Package identity and documentation
- Launcher and project-root discovery
- Command Center route and browser-rendering contracts
- Pipeline stage and route contracts
- Credential encryption/decryption paths
- Backend startup behavior
- Secret scanning
- Release documentation

## Target machine
- Windows and supported Python 3.11-3.14
- Required executables
- NVIDIA/PyTorch CUDA observability when applicable
- llama.cpp checkout and export tools
- Pipeline doctor and Command Center health

## Human-visible
- Command Center launch and browser health
- Browser rendering contract
- Dataset selection and pipeline controls
- Stop behavior
- Four credential slots and safe save/replace behavior
- Sources, Crawler, Training, Outputs, Logs, System, Configuration, Diagnostics, and Command Center

No final release claim should be made until automated, target-machine, and human-visible checks are recorded as PASS or UNKNOWN.