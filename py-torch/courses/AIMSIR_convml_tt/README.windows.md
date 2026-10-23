# Special instructions for Windows

**NB**: Your copy of Windows will need to be the 64-bit version (if you
don't know how to check that you can [find out
here](https://support.microsoft.com/en-us/windows/which-version-of-windows-operating-system-am-i-running-628bec99-476a-2c13-5296-9dd081cdd808))

1. Open PowerShell (search for "PowerShell" in the start menu) and install
   [uv](https://docs.astral.sh/uv/):

```powershell
PS> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

   Close and re-open PowerShell afterwards so that `uv` is available.

2. To "check out" (get a local copy of) the course notes you will need `git`.
   If you don't already have it you can install it with:

```powershell
PS> winget install --id Git.Git -e
```

3. Move to a suitable folder where you want to store the course material. For
   example your desktop

```powershell
PS> cd $HOME\Desktop
```

4. Check out a copy of the course material with git and move into it

```powershell
PS> git clone https://github.com/leifdenby/AIMSIR_convml_tt
PS> cd AIMSIR_convml_tt
```

5. Install `convml-tt` and everything else needed for the exercises. Use the
   `cpu` extra, unless you have an NVIDIA GPU, in which case use the extra
   matching your CUDA version (`gpu-cu118`, `gpu-cu121` or `gpu-cu124`):

```powershell
PS> uv sync --extra cpu
```

6. Open up the jupyter notebooks

```powershell
PS> uv run jupyter notebook
```
