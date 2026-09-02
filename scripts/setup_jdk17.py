"""
scripts/setup_jdk17.py
Downloads and extracts Eclipse Adoptium OpenJDK 17 LTS to resolve Java 25 / Gradle incompatibilities.
"""

import os
import sys
import zipfile
import urllib.request

DEST_DIR = r"C:\Users\Abhijeet ojha\Downloads\jdk-17"
ZIP_PATH = r"C:\Users\Abhijeet ojha\Downloads\jdk17.zip"
JDK_URL = "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.12%2B7/OpenJDK17U-jdk_x64_windows_hotspot_17.0.12_7.zip"

def main():
    if os.path.exists(DEST_DIR) and os.path.exists(os.path.join(DEST_DIR, "bin", "java.exe")):
        print(f"[+] JDK 17 already extracted at: {DEST_DIR}")
        return

    print("[*] Downloading OpenJDK 17 LTS from Eclipse Adoptium (~180MB)...")
    urllib.request.urlretrieve(JDK_URL, ZIP_PATH)
    print("[*] Extracting JDK 17...")

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(r"C:\Users\Abhijeet ojha\Downloads\temp_jdk")

    # Find the extracted folder
    extracted_roots = [d for d in os.listdir(r"C:\Users\Abhijeet ojha\Downloads\temp_jdk") if d.startswith("jdk")]
    if extracted_roots:
        inner_path = os.path.join(r"C:\Users\Abhijeet ojha\Downloads\temp_jdk", extracted_roots[0])
        if os.path.exists(DEST_DIR):
            import shutil
            shutil.rmtree(DEST_DIR)
        os.rename(inner_path, DEST_DIR)
        import shutil
        shutil.rmtree(r"C:\Users\Abhijeet ojha\Downloads\temp_jdk", ignore_errors=True)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print(f"[+] OpenJDK 17 Ready at: {DEST_DIR}")

if __name__ == "__main__":
    main()
