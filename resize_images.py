import os
try:
    from PIL import Image
    print("PIL_OK")
    src = r"D:\Capstone\assets\images"
    out_dir = os.path.join(src, "web")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(src):
        if not f.lower().endswith(".jpg"):
            continue
        path = os.path.join(src, f)
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((1200, 800))
            out = os.path.join(out_dir, f)
            im.save(out, "JPEG", quality=80, optimize=True)
            print("OK", f, os.path.getsize(out)//1024, "KB")
except ImportError as e:
    print("NO_PIL", e)