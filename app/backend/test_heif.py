from PIL import Image
import pillow_heif
import sys
import os

print(f"Python: {sys.executable}")
print(f"Pillow-heif version: {pillow_heif.__version__}")

# Register heif opener
pillow_heif.register_heif_opener()

print("Pillow-heif registered.")

try:
    print("Checking is_supported on dummy data...")
    # Should return False, not crash
    res = pillow_heif.is_supported(b'not a heic')
    print(f"is_supported returned: {res}")
except Exception as e:
    print(f"is_supported crashed: {e}")

