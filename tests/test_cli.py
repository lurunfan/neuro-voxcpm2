import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer import arguments, sha256

class CliTests(unittest.TestCase):
    def test_prompt_requires_reference(self):
        with self.assertRaises(SystemExit): arguments(['--text','hello','--style','prompt'])
    def test_empty_text_rejected(self):
        with self.assertRaises(SystemExit): arguments(['--text','  '])
    def test_default_without_reference(self):
        self.assertEqual(arguments(['--text','hello']).style, 'composed')
    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'out.wav';p.write_bytes(b'existing')
            with self.assertRaises(SystemExit): arguments(['--text','hello','--output',str(p)])
            self.assertEqual(p.read_bytes(),b'existing')
    def test_checksum_detects_change(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'weights';p.write_bytes(b'abc');before=sha256(p)
            p.write_bytes(b'abd');self.assertNotEqual(before,sha256(p))

if __name__=='__main__':unittest.main()
