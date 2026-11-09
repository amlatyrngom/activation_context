"""Compare a local tree with its S3 copy per top-level folder; optionally prune local data files.

Usage: python verify_prune.py LOCAL_DIR S3_URI [--exclude NAME ...] [--prune]
Verification: every regular file (symlinks skipped, as `aws s3 sync --no-follow-symlinks` does) must
exist on S3 with the same size. Prune (only when verification is clean): delete every file except
small text files (< 1 MB, text-like extension), then remove empty directories.
"""
import json, os, subprocess, sys
from collections import defaultdict

KEEP_EXT = {'.py', '.sh', '.md', '.json', '.txt', '.log', '.yaml', '.yml', '.toml', '.tsv', '.csv',
            '.tex', '.html', '.cfg', '.ini', '.jinja', '.patch', '.diff'}
KEEP_MAX = 1_000_000

def s3_sizes(uri):
    bucket, _, prefix = uri[5:].partition('/')
    prefix = prefix.rstrip('/') + '/'
    sizes, token = {}, None
    while True:
        cmd = ['aws', 's3api', 'list-objects-v2', '--bucket', bucket, '--prefix', prefix, '--output', 'json']
        if token:
            cmd += ['--starting-token', token]
        out = json.loads(subprocess.check_output(cmd) or b'{}')
        for o in out.get('Contents', []):
            sizes[o['Key'][len(prefix):]] = o['Size']
        token = out.get('NextToken')
        if not token:
            return sizes

def main():
    local, uri = sys.argv[1], sys.argv[2]
    excl = set(sys.argv[sys.argv.index('--exclude') + 1:]) if '--exclude' in sys.argv else set()
    excl = {e for e in excl if not e.startswith('--')}
    prune = '--prune' in sys.argv
    remote = s3_sizes(uri)
    per = defaultdict(lambda: [0, 0, 0])  # files, bytes, problems
    problems = []
    for root, dirs, files in os.walk(local):
        rel_root = os.path.relpath(root, local)
        if rel_root.split(os.sep)[0] in excl:
            dirs[:] = []
            continue
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, local)
            top = rel.split(os.sep)[0] if os.sep in rel else '.'
            size = os.path.getsize(p)
            per[top][0] += 1; per[top][1] += size
            if remote.get(rel) != size:
                per[top][2] += 1; problems.append((rel, size, remote.get(rel)))
    for top, (n, b, bad) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f'{top:40s} {n:8d} files {b/1e9:9.2f} GB  missing/mismatched: {bad}')
    print(f'TOTAL {sum(v[0] for v in per.values())} files {sum(v[1] for v in per.values())/1e9:.2f} GB; problems {len(problems)}')
    for x in problems[:20]:
        print('  PROBLEM', x)
    if not prune:
        return
    if problems:
        sys.exit('not pruning: verification found problems')
    freed = 0
    for root, dirs, files in os.walk(local):
        if os.path.relpath(root, local).split(os.sep)[0] in excl:
            dirs[:] = []
            continue
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            size = os.path.getsize(p)
            if size < KEEP_MAX and os.path.splitext(f)[1].lower() in KEEP_EXT:
                continue
            os.remove(p); freed += size
    for root, dirs, files in os.walk(local, topdown=False):
        if root != local and not os.listdir(root):
            os.rmdir(root)
    print(f'pruned {freed/1e9:.2f} GB')

if __name__ == '__main__':
    main()
