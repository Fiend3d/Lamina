from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import setup


class Response(BytesIO):
    def __init__(self, data: bytes, status: int, content_range: str = ""):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


class DownloadTest(unittest.TestCase):
    def test_short_stream_resumes_at_exact_offset(self):
        responses = [Response(b"abcde", 200), Response(b"fghij", 206, "bytes 5-9/10")]
        requests = []

        def open_request(request, timeout):
            requests.append(request)
            return responses.pop(0)

        with TemporaryDirectory() as directory:
            target = Path(directory) / "model.gguf"
            with patch.object(setup, "SIZE", 10), patch.object(setup.urllib.request, "urlopen", open_request), \
                    patch.object(setup, "validate_file"):
                setup.download(target)
            self.assertEqual(target.read_bytes(), b"abcdefghij")
            self.assertEqual(requests[1].get_header("Range"), "bytes=5-")


if __name__ == "__main__":
    unittest.main()
