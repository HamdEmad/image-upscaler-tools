import subprocess
import sys


def test_cli_list_all():
    cmd = [sys.executable, "-m", "image_upscaler_tools", "--list-all"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "[Denoisers]" in res.stdout
    assert "fast_nlm" in res.stdout
    assert "[Upscalers]" in res.stdout
    assert "span" in res.stdout
    assert "[Presets]" in res.stdout
    assert "catalog_fast" in res.stdout


def test_cli_describe():
    cmd = [sys.executable, "-m", "image_upscaler_tools", "--describe", "span"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Tool: span" in res.stdout
    assert "Category: upscaler" in res.stdout
