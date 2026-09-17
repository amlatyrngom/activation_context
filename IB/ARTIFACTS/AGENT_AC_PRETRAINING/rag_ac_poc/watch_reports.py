"""Refresh the local overview from already-synced reports; this does not monitor GPU jobs."""
import importlib.util
from pathlib import Path
import threading
spec=importlib.util.spec_from_file_location('run_poc',Path(__file__).with_name('run_poc.py'))
poc=importlib.util.module_from_spec(spec);spec.loader.exec_module(poc)
stop=threading.Event()
try:
 while True:
  poc.overview()
  if stop.wait(15):break
except KeyboardInterrupt:
 pass
