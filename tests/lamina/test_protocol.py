import sys
import threading
import unittest
from tools.lamina_protocol import NativeProcess


class CancellationTests(unittest.TestCase):
    def test_disconnect_interrupts_blocked_prompt_write(self):
        native=NativeProcess([sys.executable,"-c","import time; time.sleep(60)"],timeout=10)
        disconnected=threading.Event();native.cancelled=disconnected.is_set
        errors=[]
        def send():
            try: native.command("PROMPT 2048 " + "42 "*4000000)
            except BaseException as error: errors.append(error)
        writer=threading.Thread(target=send,daemon=True)
        try:
            writer.start()
            writer.join(timeout=0.1)
            self.assertTrue(writer.is_alive(),"fixture did not block in the native pipe")
            disconnected.set()
            writer.join(timeout=5)
            self.assertFalse(writer.is_alive(),"blocked native write survived disconnect")
            self.assertEqual(len(errors),1)
            self.assertIsInstance(errors[0],ConnectionResetError)
            self.assertIsNotNone(native.process.poll())
        finally:
            native.close()
            writer.join(timeout=1)


if __name__=="__main__":unittest.main()
