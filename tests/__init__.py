"""Test package. The whole test run writes traces into a temporary store, never into
data/traces/ (v4T02): some API tests do not use tests/api_support.py."""
import atexit
import os
import shutil
import tempfile

_TRACES_TMP = tempfile.mkdtemp(prefix="rag-test-traces-")
os.environ["TRACES_DB"] = os.path.join(_TRACES_TMP, "traces.sqlite")
atexit.register(shutil.rmtree, _TRACES_TMP, True)