"""T10 / T19: own-corpus storage, upload selection, unsupported uploads."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app import core, main

REAL_DATA = core.DATA
VELLORA = "vellora-duo-espresso-machine.txt"


class DocumentsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copytree(REAL_DATA / "corpora", self.tmp / "corpora")
        for target in (core, main):
            patcher = mock.patch.object(target, "DATA", self.tmp)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.client = TestClient(main.app)

    def names(self, corpus):
        return [d["name"] for d in core.read_corpus(corpus)]

    def test_vellora_lives_in_the_own_folder(self):
        self.assertTrue((REAL_DATA / "corpora" / "own" / VELLORA).is_file())
        self.assertEqual(self.names("own"), [VELLORA])

    def test_upload_to_own_keeps_vellora(self):
        # Before T10: the first upload replaced the whole own corpus.
        r = self.client.post("/api/documents/own", files={"file": ("notes.txt", b"hello world")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sorted(self.names("own")), sorted(["notes.txt", VELLORA]))

    def test_upload_is_added_to_existing_selection(self):
        before = json.loads((self.tmp / "corpora" / "kestrel" / "selection.json").read_text())
        r = self.client.post("/api/documents/kestrel", files={"file": ("extra.md", b"# Extra")})
        self.assertEqual(r.status_code, 200)
        after = json.loads((self.tmp / "corpora" / "kestrel" / "selection.json").read_text())
        self.assertEqual(sorted(after), sorted(before + ["extra.md"]))
        self.assertIn("extra.md", self.names("kestrel"))

    def test_delete_removes_from_selection(self):
        self.client.post("/api/documents/kestrel", files={"file": ("extra.md", b"# Extra")})
        r = self.client.delete("/api/documents/kestrel/extra.md")
        self.assertEqual(r.status_code, 200)
        after = json.loads((self.tmp / "corpora" / "kestrel" / "selection.json").read_text())
        self.assertNotIn("extra.md", after)

    def test_unsupported_upload_is_415(self):
        r = self.client.post("/api/documents/own", files={"file": ("report.pdf", b"%PDF")})
        self.assertEqual(r.status_code, 415)


if __name__ == "__main__":
    unittest.main()