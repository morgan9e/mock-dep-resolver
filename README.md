# mock-dep-resolver

Resolve both forward and reverse dependancy for mock chain builds



### Usage
```
usage: mock-dep-resolver.py [-h] [-r RELEASE] [-s SOURCE] [--download [PATH]] [--forward] [--reverse] [-v] package

Resolve deps for cross-release SRPM building via mock

positional arguments:
  package               Package name or path to .src.rpm

options:
  -h, --help            show this help message and exit
  -r, --release RELEASE
                        Target Fedora release (default: 43)
  -s, --source SOURCE   Source release (default: rawhide)
  --download [PATH]     Download SRPMs (default: ./SRPMS)
  --forward             Only resolve build deps
  --reverse             Only resolve reverse deps
  -v, --verbose
```


### Example
*For building rawhide gnome-shell for f43*

```
$ python3 mock-dep-resolver.py gnome-shell

 Package:  gnome-shell
 Target:   Fedora 43
 Source:   rawhide
 Mode:     both
 Download: no

[INFO] Loading target repo...
[INFO] Loading source repo...
[INFO] Resolving build dependencies...

[INFO] Build deps: gnome-shell
[WARN]   UNMET: mutter-devel >= 50~alpha
[INFO] Build deps: mutter

[INFO] Reverse (#1):
[WARN]   BROKEN: gnome-shell-extension-background-logo
[WARN]     requires: gnome-shell(api) = 49
[WARN]     old: gnome-shell(api) = 49
[WARN]     new: gnome-shell(api) = 50

[INFO] Forward (#1):
[INFO] Need rebuild: gnome-shell-extension-background-logo (source: gnome-shell-extension-background-logo)
[INFO] Build deps: gnome-shell-extension-background-logo

[INFO] Reverse (#2):
[OK] No more reverse dependency

====== DONE ======

[OK] Build order (3 packages):

  1. mutter
  2. gnome-shell
  3. gnome-shell-extension-background-logo

[OK] Download SRPMs:

 dnf download --source --releasever=rawhide mutter
 dnf download --source --releasever=rawhide gnome-shell
 dnf download --source --releasever=rawhide gnome-shell-extension-background-logo

[OK] Command for Mock:

mock --chain -r fedora-43-x86_64 --localrepo ~/repo \
  mutter-*.src.rpm \
  gnome-shell-*.src.rpm \
  gnome-shell-extension-background-logo-*.src.rpm

```
