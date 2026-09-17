"""Verify the reviewed sources, then run the requested experiment phase."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
root=Path(os.environ['AC_RECON_SOURCE_ROOT']);bench=Path(__file__).parent
manifest=json.loads((bench/'training_source.json').read_text())
for relative,expected in manifest['files'].items():
 path=bench/Path(relative).name if relative.startswith('IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/') else root/relative
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
print('Verified source snapshot',manifest['commit'],flush=True)
sys.exit(subprocess.call([sys.executable,str(bench/'run_recon.py'),*sys.argv[1:]]))
