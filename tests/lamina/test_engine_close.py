import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.lamina_chat import Engine  # noqa: E402


class EngineCloseTest(unittest.TestCase):
    def test_close_does_not_wait_for_a_running_request(self):
        """A request holds the engine lock for its whole answer (minutes). Ctrl+C must not wait for it."""
        engine = Engine.__new__(Engine)
        engine.lock = threading.Lock()
        engine.vision = None
        stopped = []

        class Native:
            def close(self):
                stopped.append(time.perf_counter())

        engine.native = Native()
        engine.lock.acquire()  # the request in progress
        started = time.perf_counter()
        engine.close()
        self.assertLess(time.perf_counter() - started, 3.0)
        self.assertEqual(len(stopped), 1, "the engine process must be stopped even while the lock is held")
        self.assertTrue(engine.lock.locked(), "close() must not release a lock that the request still owns")
        self.assertIsNone(engine.native)

    def test_close_when_idle(self):
        engine = Engine.__new__(Engine)
        engine.lock = threading.Lock()
        engine.vision = None
        engine.native = None
        engine.close()
        self.assertFalse(engine.lock.locked())


if __name__ == "__main__":
    unittest.main()
