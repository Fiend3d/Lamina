import subprocess
import unittest
from unittest.mock import patch

from tools import lamina_hardware


class HardwareTest(unittest.TestCase):
    def test_gpu_from_nvidia_smi(self):
        reply = subprocess.CompletedProcess([], 0, stdout="NVIDIA GeForce RTX 3050, 8192\n", stderr="")
        with patch("tools.lamina_hardware.subprocess.run", return_value=reply):
            self.assertEqual(lamina_hardware.gpu(), "NVIDIA GeForce RTX 3050 8 GB")

    def test_missing_nvidia_smi(self):
        with patch("tools.lamina_hardware.subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(lamina_hardware.gpu(), "unknown GPU")

    def test_describe_has_gpu_cpu_and_ram(self):
        with patch("tools.lamina_hardware.gpu", return_value="GPU"), \
             patch("tools.lamina_hardware.cpu", return_value="CPU"), \
             patch("tools.lamina_hardware.ram_gib", return_value=63.9):
            self.assertEqual(lamina_hardware.describe(), "GPU / CPU / 64 GB RAM")


if __name__ == "__main__":
    unittest.main()
