from gemini_client import analyze_image, get_provider_status
from PIL import Image, ImageDraw
import io, json

# Create test image
img = Image.new("RGB", (400, 200), "white")
draw = ImageDraw.Draw(img)
draw.text((20, 30), "What is 2 + 2?", fill="black")
draw.text((20, 70), "A) 3", fill="black")
draw.text((20, 95), "B) 4", fill="black")
draw.text((20, 120), "C) 5", fill="black")
draw.text((20, 145), "D) 6", fill="black")
buf = io.BytesIO()
img.save(buf, format="JPEG")
img_bytes = buf.getvalue()

# Test auto-fallback mode
print("=== Testing auto-fallback mode ===")
try:
    result = analyze_image(img_bytes, mode="auto")
    print(f"Provider: {result.get('provider', 'unknown')}")
    print(f"Detected: {result.get('detected')}")
    print(f"Type: {result.get('type')}")
    print(f"Answer: {result.get('answer')}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Provider Status ===")
status = get_provider_status()
for key, info in status.items():
    avail = "READY" if info["available"] else f"COOLDOWN ({info['cooldown_remaining']}s)"
    print(f"  {info['name']}: {avail} | model={info['model']}")
