import os, hashlib

def make_id(prefix: str, name: str) -> str:
    h = hashlib.sha1(name.encode()).hexdigest()[:10]
    return f"{prefix}_{h}"

def save_uploaded_file(uploaded, target_dir: str) -> str:
    path = os.path.join(target_dir, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path
