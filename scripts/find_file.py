import os

print("Searching for yabgu_pkg.zip...")
found = []
for root, dirs, files in os.walk("/"):
    if "yabgu_pkg.zip" in files:
        found.append(os.path.join(root, "yabgu_pkg.zip"))
    # skip deep system dirs
    if any(skip in root for skip in ["/proc", "/sys", "/usr", "/lib", "/var"]):
        dirs.clear()

print("Found files:", found)
