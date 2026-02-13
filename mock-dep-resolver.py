#!/usr/bin/env python3
"""
mock-dep-resolver.py - Resolve deps for cross-release SRPM building via mock

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
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

try:
    import rpm
except ImportError:
    sys.exit("Error: python3-rpm required (sudo dnf install python3-rpm)")
try:
    import dnf
except ImportError:
    sys.exit("Error: python3-dnf required (sudo dnf install python3-dnf)")


class C:
    R  = '\033[0;31m' if sys.stderr.isatty() else ''
    G  = '\033[0;32m' if sys.stderr.isatty() else ''
    Y  = '\033[1;33m' if sys.stderr.isatty() else ''
    B  = '\033[0;34m' if sys.stderr.isatty() else ''
    CN = '\033[0;36m' if sys.stderr.isatty() else ''
    DIM = '\033[2m'   if sys.stderr.isatty() else ''
    N  = '\033[0m'    if sys.stderr.isatty() else ''

def log(m):   print(f"{C.B}[INFO]{C.N} {m}", file=sys.stderr)
def warn(m):  print(f"{C.Y}[WARN]{C.N} {m}", file=sys.stderr)
def err(m):   print(f"{C.R}[ERROR]{C.N} {m}", file=sys.stderr)
def ok(m):    print(f"{C.G}[OK]{C.N} {m}", file=sys.stderr)
def debug(m, v):
    if v: print(f"  {C.B}[·]{C.N} {m}", file=sys.stderr)  # noqa: E701


def sourcerpm_to_name(s):
    return re.sub(r'-[^-]+-[^-]+\.src\.rpm$', '', s) if s else None


### Repo querier ###

class RepoQuerier:
    def __init__(self, releasever, enable_source=False, verbose=False):
        self.releasever = str(releasever)
        self.enable_source = enable_source
        self.verbose = verbose
        self._base = None

    def _add_repos(self, base):
        arch = platform.machine()
        if self.releasever == 'rawhide':
            self._add_repo(base, 'rawhide',
                f'https://mirrors.fedoraproject.org/metalink?repo=rawhide&arch={arch}')
            if self.enable_source:
                self._add_repo(base, 'rawhide-source',
                    f'https://mirrors.fedoraproject.org/metalink?repo=rawhide-source&arch={arch}')
        else:
            rv = self.releasever
            self._add_repo(base, 'fedora',
                f'https://mirrors.fedoraproject.org/metalink?repo=fedora-{rv}&arch={arch}')
            self._add_repo(base, 'updates',
                f'https://mirrors.fedoraproject.org/metalink?repo=updates-released-f{rv}&arch={arch}')
            if self.enable_source:
                self._add_repo(base, 'fedora-source',
                    f'https://mirrors.fedoraproject.org/metalink?repo=fedora-source-{rv}&arch={arch}')
                self._add_repo(base, 'updates-source',
                    f'https://mirrors.fedoraproject.org/metalink?repo=updates-released-source-f{rv}&arch={arch}')

    @staticmethod
    def _add_repo(base, repo_id, metalink):
        repo = dnf.repo.Repo(repo_id, base.conf)
        repo.metalink = metalink
        repo.gpgcheck = False
        repo.skip_if_unavailable = True
        base.repos.add(repo)
        repo.enable()

    def _get_base(self):
        if self._base is None:
            debug(f"Loading repos for releasever={self.releasever} "
                f"(source={'yes' if self.enable_source else 'no'})", self.verbose)
            b = dnf.Base()
            b.conf.releasever = self.releasever
            b.conf.cachedir = os.path.join(
                tempfile.gettempdir(), f'mock-dep-resolver-{self.releasever}')
            b.conf.substitutions['releasever'] = self.releasever
            self._add_repos(b)
            enabled = [r.id for r in b.repos.iter_enabled()]
            debug(f"Enabled repos: {', '.join(enabled)}", self.verbose)
            b.fill_sack(load_system_repo=False)
            self._base = b
        return self._base

    def get_build_requires(self, pkg_name):
        base = self._get_base()
        q = base.sack.query().available().filter(name=pkg_name, arch='src')
        pkgs = list(q)
        if not pkgs:
            return None
        reqs = []
        for r in pkgs[-1].requires:
            s = str(r)
            if not s.startswith('rpmlib('):
                reqs.append(s)
        return reqs

    def is_dep_satisfied(self, dep_str):
        base = self._get_base()
        return len(base.sack.query().available().filter(provides=dep_str)) > 0

    def dep_to_source(self, dep_str):
        base = self._get_base()
        for pkg in base.sack.query().available().filter(provides=dep_str):
            if pkg.sourcerpm:
                return sourcerpm_to_name(pkg.sourcerpm)
        return None

    def binary_to_source(self, pkg_name):
        base = self._get_base()
        for pkg in base.sack.query().available().filter(name=pkg_name):
            if pkg.sourcerpm:
                return sourcerpm_to_name(pkg.sourcerpm)
        return None

    def get_provides(self, pkg_name):
        base = self._get_base()
        provs = []
        for pkg in base.sack.query().available().filter(name=pkg_name):
            for p in pkg.provides:
                provs.append(str(p))
        return provs

    def source_to_binaries(self, src_name):
        base = self._get_base()
        bins = set()
        for pkg in base.sack.query().available():
            if pkg.sourcerpm and sourcerpm_to_name(pkg.sourcerpm) == src_name:
                bins.add(pkg.name)
        return bins

    def close(self):
        if self._base:
            self._base.close()
            self._base = None


### Local RPM helpers ###

def get_srpm_build_requires(srpm_path):
    ts = rpm.TransactionSet()
    ts.setVSFlags(rpm._RPMVSF_NOSIGNATURES | rpm._RPMVSF_NODIGESTS)
    with open(srpm_path, 'rb') as f:
        hdr = ts.hdrFromFdno(f)
    name = hdr[rpm.RPMTAG_NAME]
    if isinstance(name, bytes):
        name = name.decode()
    reqs = []
    rn = hdr[rpm.RPMTAG_REQUIRENAME]
    rf = hdr[rpm.RPMTAG_REQUIREFLAGS]
    rv = hdr[rpm.RPMTAG_REQUIREVERSION]
    if rn:
        for i, n in enumerate(rn):
            if isinstance(n, bytes):
                n = n.decode()
            if n.startswith('rpmlib('):
                continue
            dep = n
            if rf and rv and rv[i]:
                v = rv[i]
                if isinstance(v, bytes):
                    v = v.decode()
                if v:
                    fl = rf[i]
                    op = ''
                    if fl & rpm.RPMSENSE_LESS:
                        op += '<'
                    if fl & rpm.RPMSENSE_GREATER:
                        op += '>'
                    if fl & rpm.RPMSENSE_EQUAL:
                        op += '='
                    if op: dep = f"{n} {op} {v}"
            reqs.append(dep)
    return name, reqs


def get_installed_provides(pkg_name):
    ts = rpm.TransactionSet()
    provs = []
    for hdr in ts.dbMatch('name', pkg_name):
        pn = hdr[rpm.RPMTAG_PROVIDENAME]
        pf = hdr[rpm.RPMTAG_PROVIDEFLAGS]
        pv = hdr[rpm.RPMTAG_PROVIDEVERSION]
        if pn:
            for i, n in enumerate(pn):
                if isinstance(n, bytes):
                    n = n.decode()
                s = n
                if pf and pv and pv[i]:
                    v = pv[i]
                    if isinstance(v, bytes):
                        v = v.decode()
                    if v:
                        fl = pf[i]
                        op = ''
                        if fl & rpm.RPMSENSE_LESS:
                            op += '<'
                        if fl & rpm.RPMSENSE_GREATER:
                            op += '>'
                        if fl & rpm.RPMSENSE_EQUAL:
                            op += '='
                        if op: s = f"{n} {op} {v}"
                provs.append(s)
    return provs


def get_installed_reverse_deps(provide_name):
    r = subprocess.run(['rpm', '-q', '--whatrequires', provide_name],
                       capture_output=True, text=True)
    pkgs = set()
    for line in r.stdout.strip().splitlines():
        if line.startswith('no package'):
            continue
        n = subprocess.run(['rpm', '-q', '--qf', '%{NAME}', line],
                           capture_output=True, text=True).stdout.strip()
        if n:
            pkgs.add(n)
    return pkgs


def get_installed_requires(pkg_name):
    ts = rpm.TransactionSet()
    reqs = []
    for hdr in ts.dbMatch('name', pkg_name):
        rn = hdr[rpm.RPMTAG_REQUIRENAME]
        rf = hdr[rpm.RPMTAG_REQUIREFLAGS]
        rv = hdr[rpm.RPMTAG_REQUIREVERSION]
        if rn:
            for i, n in enumerate(rn):
                if isinstance(n, bytes):
                    n = n.decode()
                if n.startswith('rpmlib('):
                    continue
                v, op = '', ''
                if rf and rv and rv[i]:
                    v = rv[i]
                    if isinstance(v, bytes):
                        v = v.decode()
                    if v:
                        fl = rf[i]
                        if fl & rpm.RPMSENSE_LESS:
                            op += '<'
                        if fl & rpm.RPMSENSE_GREATER:
                            op += '>'
                        if fl & rpm.RPMSENSE_EQUAL:
                            op += '='
                reqs.append((n, op, v))
    return reqs


def is_pkg_installed(pkg_name):
    ts = rpm.TransactionSet()
    return ts.dbMatch('name', pkg_name).count() > 0


def download_srpm(pkg_name, source_release, workdir, verbose=False):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for f in workdir.glob(f"{pkg_name}-*.src.rpm"):
        debug(f"Already have: {f.name}", verbose)
        return str(f)
    log(f"Downloading SRPM: {pkg_name}")
    r = subprocess.run(
        ['dnf', 'download', '--source', f'--releasever={source_release}',
         '--destdir', str(workdir), pkg_name],
        capture_output=True, text=True)
    if r.returncode != 0:
        err(f"Failed to download {pkg_name}: {r.stderr.strip()}")
        return None
    for f in sorted(workdir.glob(f"{pkg_name}-*.src.rpm"),
                    key=os.path.getmtime, reverse=True):
        return str(f)
    return None


### Forward resolution ###

def resolve_forward(pkg_name, srpm_path, target_repo, source_repo,
                    source_release, download_dir, verbose,
                    visited, build_order, all_to_build, dep_children):
    if pkg_name in visited:
        return
    visited.add(pkg_name)
    dep_children.setdefault(pkg_name, [])

    if srpm_path and os.path.isfile(srpm_path):
        _, breqs = get_srpm_build_requires(srpm_path)
    else:
        breqs = source_repo.get_build_requires(pkg_name)
        if breqs is None:
            err(f"Cannot find source package '{pkg_name}' in {source_release}")
            return

    log(f"Build deps: {pkg_name}")

    for dep in breqs:
        if target_repo.is_dep_satisfied(dep):
            debug(f"ok: {dep}", verbose)
        else:
            warn(f"  UNMET: {dep}")
            src = source_repo.dep_to_source(dep)
            if not src:
                err(f"  Cannot find source for: {dep}")
                continue
            debug(f"  -> source: {src}", verbose)
            if src not in [c for c, _ in dep_children[pkg_name]]:
                dep_children[pkg_name].append((src, "build"))
            if src in visited:
                continue
            dep_srpm = None
            if download_dir:
                dep_srpm = download_srpm(src, source_release, download_dir, verbose)
            resolve_forward(src, dep_srpm, target_repo, source_repo,
                            source_release, download_dir, verbose,
                            visited, build_order, all_to_build, dep_children)
            all_to_build[src] = dep_srpm
            if not any(n == src for n, _ in build_order):
                build_order.append((src, dep_srpm))

    all_to_build[pkg_name] = srpm_path
    if not any(n == pkg_name for n, _ in build_order):
        build_order.append((pkg_name, srpm_path))


### Reverse resolution ###

def resolve_reverse(all_to_build, source_repo, visited, verbose):
    newly_broken = []
    checked = set()

    for src_name in list(all_to_build.keys()):
        binaries = source_repo.source_to_binaries(src_name)
        binaries.add(src_name)

        for bin_pkg in binaries:
            if bin_pkg in checked:
                continue
            checked.add(bin_pkg)

            if not is_pkg_installed(bin_pkg):
                debug(f"revdep: {bin_pkg} not installed", verbose)
                continue

            debug(f"revdep: checking {bin_pkg}", verbose)

            old_provs = set(get_installed_provides(bin_pkg))
            new_provs = set(source_repo.get_provides(bin_pkg))
            if not old_provs or not new_provs:
                continue

            removed = old_provs - new_provs
            debug(f"  {len(old_provs)} old, {len(new_provs)} new, {len(removed)} removed", verbose)

            for old_prov in removed:
                prov_name = re.split(r'\s*[<>=]', old_prov)[0].strip()
                rdeps = get_installed_reverse_deps(prov_name)
                for rdep_name in rdeps:
                    if rdep_name in visited:
                        continue
                    rdep_src = source_repo.binary_to_source(rdep_name)
                    if rdep_src and rdep_src in visited:
                        continue
                    for rn, op, ver in get_installed_requires(rdep_name):
                        if rn == prov_name and op == '=' and ver:
                            req_str = f"{rn} = {ver}"
                            if req_str not in new_provs:
                                new_match = [p for p in new_provs if p.startswith(f"{prov_name} ")]
                                warn(f"  BROKEN: {rdep_name}")
                                warn(f"    requires: {req_str}")
                                warn(f"    old: {old_prov}")
                                warn(f"    new: {new_match[0] if new_match else '(removed)'}")
                                newly_broken.append((rdep_name, bin_pkg))

            old_so = {p for p in old_provs if '.so' in p}
            new_so = {p for p in new_provs if '.so' in p}
            for libname in (old_so - new_so):
                debug(f"  removed lib: {libname}", verbose)
                so_base = libname.split('(')[0]
                rdeps = get_installed_reverse_deps(so_base)
                for rdep_name in rdeps:
                    if rdep_name in visited:
                        continue
                    rdep_src = source_repo.binary_to_source(rdep_name)
                    if rdep_src and rdep_src in visited:
                        continue
                    if any(rdep_name == n for n, _ in newly_broken):
                        continue
                    warn(f"  BROKEN (lib): {rdep_name} requires {libname}")
                    newly_broken.append((rdep_name, bin_pkg))

    seen = set()
    unique = []
    for name, parent in newly_broken:
        if name not in seen:
            seen.add(name)
            unique.append((name, parent))
    return unique


### Output ###

def print_results(build_order, mock_config, dep_children, root_name, source_release):
    print(file=sys.stderr)
    print("====== DONE ======", file=sys.stderr)

    seen = set()
    final = []
    for name, path in build_order:
        if name not in seen:
            seen.add(name)
            final.append((name, path))

    if not final:
        ok("Nothing to build.")
        return

    if len(final) == 1 and final[0][1]:
        ok("No dependency found:")
        print(f"  mock -r {mock_config} {final[0][1]}", file=sys.stderr)
        print(final[0][1])
        return

    print(file=sys.stderr)
    ok(f"Build order ({len(final)} packages):")
    print(file=sys.stderr)
    for i, (name, path) in enumerate(final, 1):
        label = os.path.basename(path) if path else name
        print(f"  {i}. {C.CN}{label}{C.N}", file=sys.stderr)

    print(file=sys.stderr)
    ok("Download SRPMs:")
    print(file=sys.stderr)
    for name, _ in final:
        print(f" dnf download --source --releasever={source_release} {name}", file=sys.stderr)

    print(file=sys.stderr)
    ok("Command for Mock:")
    print(file=sys.stderr)
    parts = [f"mock --chain -r {mock_config} --localrepo ~/repo"]
    for name, path in final:
        entry = path if path else f"{name}-*.src.rpm"
        parts.append(f"  {entry}")
    print(" \\\n".join(parts), file=sys.stderr)
    print(file=sys.stderr)


### Main ###

def main():
    p = argparse.ArgumentParser(description='Resolve deps for cross-release SRPM building via mock')
    p.add_argument('package', help='Package name or path to .src.rpm')
    p.add_argument('-r', '--release', default='43', help='Target Fedora release (default: 43)')
    p.add_argument('-s', '--source', default='rawhide', help='Source release (default: rawhide)')
    p.add_argument('--download', nargs='?', const='./SRPMS', default=None,
                   metavar='PATH', help='Download SRPMs (default: ./SRPMS)')
    p.add_argument('--forward', action='store_true', help='Only resolve build deps')
    p.add_argument('--reverse', action='store_true', help='Only resolve reverse deps')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    do_fwd = True
    do_rev = True
    if args.forward or args.reverse:
        do_fwd = args.forward
        do_rev = args.reverse

    srpm_path = None
    if os.path.isfile(args.package) and args.package.endswith('.src.rpm'):
        srpm_path = os.path.realpath(args.package)
        pkg_name, _ = get_srpm_build_requires(srpm_path)
    else:
        pkg_name = args.package

    mock_config = f"fedora-{args.release}-x86_64"

    print(file=sys.stderr)
    sfx = f" ({os.path.basename(srpm_path)})" if srpm_path else ""
    print(f" Package:  {pkg_name}{sfx}", file=sys.stderr)
    print(f" Target:   Fedora {args.release}", file=sys.stderr)
    print(f" Source:   {args.source}", file=sys.stderr)
    mode = 'forward' if do_fwd and not do_rev else 'reverse' if do_rev and not do_fwd else 'both'
    print(f" Mode:     {mode}", file=sys.stderr)
    print(f" Download: {args.download if args.download else 'no'}", file=sys.stderr)
    print(file=sys.stderr)

    log("Loading target repo...")
    target_repo = RepoQuerier(args.release, enable_source=False, verbose=args.verbose)
    log("Loading source repo...")
    source_repo = RepoQuerier(args.source, enable_source=True, verbose=args.verbose)

    visited = set()
    build_order = []
    all_to_build = OrderedDict()
    dep_children = {}  # name -> [(child_name, "build"|"revdep")]

    if do_fwd:
        log("Resolving build dependencies...")
        print(file=sys.stderr)
        resolve_forward(pkg_name, srpm_path, target_repo, source_repo,
                        args.source, args.download, args.verbose,
                        visited, build_order, all_to_build, dep_children)

    if not do_rev:
        print_results(build_order, mock_config, dep_children, pkg_name, args.source)
        target_repo.close()
        source_repo.close()
        return

    if not do_fwd:
        all_to_build[pkg_name] = srpm_path
        visited.add(pkg_name)
        dep_children.setdefault(pkg_name, [])

    for iteration in range(1, 21):
        print(file=sys.stderr)
        log(f"Reverse (#{iteration}):")

        broken = resolve_reverse(all_to_build, source_repo, visited, args.verbose)

        if not broken:
            ok("No more reverse dependency")
            break

        print(file=sys.stderr)
        log(f"Forward (#{iteration}):")
        added = False
        for bbin, parent_bin in broken:
            bsrc = source_repo.binary_to_source(bbin)
            if not bsrc:
                err(f"Cannot find source for: {bbin}")
                continue
            if bsrc in visited:
                continue
            log(f"Need rebuild: {bbin} (source: {bsrc})")
            added = True

            parent_src = source_repo.binary_to_source(parent_bin) or parent_bin
            dep_children.setdefault(parent_src, [])
            if bsrc not in [c for c, _ in dep_children[parent_src]]:
                dep_children[parent_src].append((bsrc, "revdep"))

            dep_srpm = None
            if args.download:
                dep_srpm = download_srpm(bsrc, args.source, args.download, args.verbose)
            if do_fwd:
                resolve_forward(bsrc, dep_srpm, target_repo, source_repo,
                                args.source, args.download, args.verbose,
                                visited, build_order, all_to_build, dep_children)
            else:
                visited.add(bsrc)
                all_to_build[bsrc] = dep_srpm
                build_order.append((bsrc, dep_srpm))

        if not added:
            ok("Done")
            break
    else:
        err("Too many iterations (20).")

    print_results(build_order, mock_config, dep_children, pkg_name, args.source)
    target_repo.close()
    source_repo.close()


if __name__ == '__main__':
    main()
