#!/bin/bash
#SBATCH --job-name=msbxprobe
#SBATCH --cpus-per-task=2 --mem=8G --time=00:20:00
# Can an ORCD compute node run the agent sandbox (ROOT_CANON: rootless Podman)? Read-only probe for the math dev_hard solver runs
# (2026-11-18): container runtimes on PATH / in modules, user namespaces, subuid, fuse-overlayfs, cgroups. Nothing is pulled or run.
export PATH=/usr/local/bin:/usr/bin:/bin
date -u; hostname; cat /etc/os-release 2>/dev/null | head -2
for b in podman apptainer singularity docker fuse-overlayfs slirp4netns newuidmap bwrap; do printf '%-16s %s\n' $b "$(command -v $b || echo -)"; done
set +u; [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; module avail 2>&1 | tr -s ' ' '\n' | grep -i -E 'podman|apptainer|singularity|container' | head
echo "user.max_user_namespaces $(cat /proc/sys/user/max_user_namespaces 2>/dev/null)"
echo "subuid: $(grep -c "^$(id -un):" /etc/subuid 2>/dev/null) subgid: $(grep -c "^$(id -un):" /etc/subgid 2>/dev/null)"
echo "cgroup: $(stat -fc %T /sys/fs/cgroup 2>/dev/null)"; ls -ld /dev/fuse 2>&1
command -v podman >/dev/null && podman info --format '{{.Host.OCIRuntime.Name}} {{.Store.GraphDriverName}} rootless={{.Host.Security.Rootless}}' 2>&1 | tail -3
command -v apptainer >/dev/null && apptainer --version
command -v unshare >/dev/null && unshare -Ur true 2>&1 && echo "unshare -Ur: ok"
