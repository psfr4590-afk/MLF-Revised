# First boot on a new Windows laptop

ML-Framework supports two legitimate ways to arrive at the project:

1. **GitHub ZIP download** for someone who is new to Git.
2. **Git clone** for a developer who already uses Git.

The ZIP path is intentionally supported. A clean machine is not assumed to have Git installed.

## ZIP path

After downloading the repository with GitHub's **Code → Download ZIP** and extracting it:

1. Open PowerShell in the extracted `ML-Framework` folder.
2. Run:

~~~
.\bootstrap_windows.ps1
~~~

The Windows first-boot script:

- locates a supported Python 3.11-3.14 interpreter;
- if Python is missing and Windows `winget` is available, installs Python 3.13 for the current user;
- runs the canonical `bootstrap.py --install` dependency installation;
- selects the CPU or NVIDIA CUDA PyTorch wheel automatically;
- runs `bootstrap.py --doctor`;
- stops with an actionable message instead of continuing after a failed prerequisite.

Then prove the installation:

~~~
python .\mlframework.py smoke
~~~

Launch the localhost browser Command Center:

~~~
python .\launch.py
~~~

The launcher delegates to `run_command_center.py`; `ui/app.py` remains a compatibility launcher only.

## Native export path

Native GGUF export needs CMake and the pinned llama.cpp toolchain. If you want the first-boot script to prepare that path too:

~~~
.\bootstrap_windows.ps1 -Native
~~~

If CMake is missing and `winget` is available, the script installs the Kitware CMake package for the current user before invoking the native bootstrap.

The native path is intentionally separate from the basic smoke path. A user should not need a compiler toolchain just to prove that the Python pipeline works.

## If the machine has no winget

The script does not pretend every Windows installation is identical. If `winget` is unavailable, it reports the missing prerequisite and stops.

Install Python 3.11-3.14 manually, open a new PowerShell window, and rerun:

~~~
.\bootstrap_windows.ps1
~~~

For native export, install CMake and rerun:

~~~
.\bootstrap_windows.ps1 -Native
~~~

## Recovery behavior

The first-boot path is safe to rerun. If package installation is interrupted or the network fails, rerun the same command after fixing the reported problem. Pip will reconcile the declared dependencies rather than requiring a manual dependency inventory.

Do not delete the repository or generated provenance merely because the first installation attempt failed. The failure is an environment problem until the doctor or release verification provides evidence otherwise.

## Developer path

Git users can continue to use:

~~~
git clone https://github.com/psfr4590-afk/ML-Framework.git
cd ML-Framework
~~~

Then use the same canonical Python bootstrap:

~~~
python .\bootstrap.py --install
python .\bootstrap.py --doctor
python .\mlframework.py smoke
~~~

The two entry paths converge on the same `bootstrap.py`; the Windows wrapper exists to handle the human and machine prerequisites that occur before the Python bootstrap can even start.
