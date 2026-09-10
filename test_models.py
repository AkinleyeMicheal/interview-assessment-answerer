import json, urllib.request
req = urllib.request.Request("https://openrouter.ai/api/v1/models")
resp = urllib.request.urlopen(req, timeout=15)
data = json.loads(resp.read().decode("utf-8"))
models = data.get("data", [])
free = [m for m in models if m.get("id", "").endswith(":free")]
for m in free[:25]:
    arch = m.get("architecture", {})
    modality = arch.get("modality", "unknown")
    print(f"{m['id']}  modality={modality}  ctx={m.get('context_length', '?')}")
