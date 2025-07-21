import os
import subprocess
from datetime import datetime

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

def auto_push(commit_msg=None):
    print("📦 Staging changes...")
    run_command("git add .")

    msg = commit_msg or f"Auto-push on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    print(f"📝 Committing with message: {msg}")
    run_command(f'git commit -m "{msg}"')

    print("🚀 Pushing to origin main...")
    run_command("git push origin login-version")

if __name__ == "__main__":
    auto_push()
