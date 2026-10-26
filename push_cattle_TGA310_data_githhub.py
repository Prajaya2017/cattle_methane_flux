# -*- coding: utf-8 -*-
"""
push_cattle_TGA310_data_githhub.py

Copies the LoggerNet AmrFlux table into the local git repo and pushes ONLY the
files the Render dashboard needs (listed in PUSH_PATHS) to GitHub when they change.
Anything else in the folder (Claude outputs, previews, logs, test files, ...) is
never committed or pushed.

Usage:
    python push_cattle_TGA310_data_githhub.py            # run once (good for Task Scheduler)
    python push_cattle_TGA310_data_githhub.py --watch    # keep running, check every POLL_MINUTES

@author: pprajapati
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import time
from datetime import datetime

# =========================
# PATH SETTINGS
# =========================
# LoggerNet output file
SOURCE_FILE = r"C:\Campbellsci\LoggerNet\Data\Cattle_Experiment_Eagle_TGA310_AmrFlux.dat"

# =========================
# GIT SETTINGS
# =========================
# GitHub repo the data is pushed to
GIT_REMOTE_URL = "https://github.com/Prajaya2017/cattle_methane_flux.git"

# Local clone of that repo (created automatically if it doesn't exist)
GIT_REPO_DIR = r"C:\cattle_methane_flux"

# Path of the file inside the repo
GIT_FILE_REL_PATH = "Cattle_Experiment_Eagle_TGA310_AmrFlux.dat"

# Branch to push to
GIT_BRANCH = "main"

# Check interval for --watch mode (AmrFlux table writes every 30 min)
POLL_MINUTES = 5

# ONLY these files / folders are committed and pushed (paths relative to GIT_REPO_DIR).
# Everything else in the folder stays local. Add a path here if the dashboard needs a new file.
PUSH_PATHS = [
    GIT_FILE_REL_PATH,              # data for the dashboard
    "app.py",
    "requirements.txt",
    "render.yaml",
    "README.md",
    ".gitignore",
    "assets/theme.css",
    "assets/title_tooltip.js",
    "assets/save_pdf.js",
    "assets/wd_dropdown.js",
    "assets/jspdf.umd.min.js",
    "assets/setup",                 # Site and Setup page (setup.html + photos)
]


def file_hash(path: str) -> str:
    """MD5 of a file, or '' if it doesn't exist."""
    if not os.path.isfile(path):
        return ""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: str, dst: str, retries: int = 5, wait_s: float = 3.0) -> None:
    """Copy src -> dst, retrying in case LoggerNet has the file locked mid-write."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".tmp"
    for attempt in range(1, retries + 1):
        try:
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            return
        except (PermissionError, OSError) as e:
            print(f"[COPY] Attempt {attempt}/{retries} failed: {e}")
            time.sleep(wait_s)
    raise RuntimeError(f"Could not copy {src} after {retries} attempts")


def last_timestamp(path: str) -> str:
    """Return TIMESTAMP of the last data record (TOA5 format), for the commit message."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8192))
            lines = [ln for ln in f.read().decode("utf-8", "ignore").splitlines() if ln.strip()]
        return lines[-1].split(",")[0].strip('"') if lines else ""
    except Exception:
        return ""


def git_push_if_changed(repo_dir: str, message: str, branch: str = "main") -> None:
    """
    Stage/commit/push only the files in PUSH_PATHS if any of them changed
    (new, modified or deleted). Other files in the folder are left alone.
    Assumes:
      - repo_dir is a valid git repo
      - remote 'origin' exists
      - auth is already configured (PAT via credential manager or SSH)
    """
    repo_dir = os.path.abspath(repo_dir)
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"GIT_REPO_DIR not found: {repo_dir}")

    def run(cmd):
        subprocess.run(cmd, cwd=repo_dir, check=True)

    # Make sure branch exists/checked out
    run(["git", "checkout", branch])

    # Allowed paths that exist locally or are already tracked (so deletions are also pushed)
    tracked = set(subprocess.run(["git", "ls-files"], cwd=repo_dir, capture_output=True,
                                 text=True).stdout.split("\n"))
    paths = [p for p in PUSH_PATHS
             if os.path.exists(os.path.join(repo_dir, p))
             or p in tracked or any(t.startswith(p.rstrip("/") + "/") for t in tracked)]
    if not paths:
        print("[GIT] None of PUSH_PATHS found. Skipping push.")
        return

    # Stage only the allowed paths (new, modified and deleted)
    run(["git", "add", "-A", "--", *paths])

    # If nothing staged in the allowed paths, skip commit/push
    diff = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=repo_dir)
    if diff.returncode == 0:
        print("[GIT] No changes to commit. Skipping push.")
        return

    # Show what is being pushed
    changed = subprocess.run(["git", "diff", "--cached", "--name-status", "--", *paths], cwd=repo_dir,
                             capture_output=True, text=True).stdout.strip()
    print("[GIT] Changes to push:\n" + changed)

    # Commit, pick up any commits made on GitHub meanwhile, then push
    run(["git", "commit", "-m", message, "--", *paths])     # commit ONLY the allowed paths
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", branch], cwd=repo_dir)
    run(["git", "push", "origin", branch])

    print(f"[GIT] Pushed dashboard files -> origin/{branch}")


def ensure_gitignore(repo_dir: str) -> None:
    """Keep temp/cache files out of 'git add -A' (only created if missing)."""
    path = os.path.join(repo_dir, ".gitignore")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("__pycache__/\n*.pyc\n*.tmp\n.ipynb_checkpoints/\n.spyproject/\n"
                    ".DS_Store\nThumbs.db\ndesktop.ini\n")
        print("[GIT] Created .gitignore")
    # Folders / files that must never be pushed (added once if missing)
    with open(path) as f:
        have = f.read().split("\n")
    extra = [e for e in ("/Claude outputs", "*.log", "*.bak") if e not in have]
    if extra:
        with open(path, "a") as f:
            f.write("\n" + "\n".join(extra) + "\n")
        print(f"[GIT] Added to .gitignore: {', '.join(extra)}")


def ensure_repo(repo_dir: str, remote_url: str, branch: str) -> None:
    """
    Make sure repo_dir is a clone of remote_url with 'origin' pointing to it.
      - folder missing/empty  -> git clone
      - folder exists, no git -> git init + add origin
      - origin points elsewhere -> reset origin URL to remote_url
    """
    def git(*args, check=True):
        return subprocess.run(["git", *args], cwd=repo_dir, check=check,
                              capture_output=True, text=True)

    if not os.path.isdir(repo_dir) or not os.listdir(repo_dir):
        print(f"[GIT] Cloning {remote_url} -> {repo_dir}")
        subprocess.run(["git", "clone", remote_url, repo_dir], check=True)
    elif not os.path.isdir(os.path.join(repo_dir, ".git")):
        print(f"[GIT] Initialising repo in {repo_dir}")
        git("init")
        git("remote", "add", "origin", remote_url)
        git("fetch", "origin", check=False)

    # Point origin at the right repo
    current = git("remote", "get-url", "origin", check=False)
    if current.returncode != 0:
        git("remote", "add", "origin", remote_url)
    elif current.stdout.strip() != remote_url:
        print(f"[GIT] origin was {current.stdout.strip()} -> setting to {remote_url}")
        git("remote", "set-url", "origin", remote_url)

    # Make sure the branch exists locally (tracks origin/branch if it's on GitHub)
    if git("rev-parse", "--verify", branch, check=False).returncode != 0:
        if git("rev-parse", "--verify", f"origin/{branch}", check=False).returncode == 0:
            git("checkout", "-b", branch, "--track", f"origin/{branch}")
        else:
            # Empty GitHub repo: create the branch with a first commit so
            # 'git checkout <branch>' in git_push_if_changed works
            git("checkout", "-B", branch)
            readme = os.path.join(repo_dir, "README.md")
            if not os.path.exists(readme):
                with open(readme, "w") as f:
                    f.write("# cattle_methane_flux\n")
            git("add", "README.md")
            git("commit", "-m", "first commit")
            print(f"[GIT] Created branch '{branch}' with first commit")

    # Tell git to link local branch to origin/branch on first push
    git("config", f"branch.{branch}.remote", "origin")
    git("config", f"branch.{branch}.merge", f"refs/heads/{branch}")


def sync_once() -> None:
    """Copy the AmrFlux table into the repo and push the dashboard files if they changed."""
    if not os.path.isfile(SOURCE_FILE):
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    ensure_repo(GIT_REPO_DIR, GIT_REMOTE_URL, GIT_BRANCH)
    ensure_gitignore(GIT_REPO_DIR)

    repo_target = os.path.join(GIT_REPO_DIR, GIT_FILE_REL_PATH)

    # Always refresh the repo copy; git_push_if_changed() decides (via git diff)
    # whether there is anything new to commit/push. This also retries a push
    # that failed on a previous run.
    if file_hash(SOURCE_FILE) != file_hash(repo_target):
        safe_copy(SOURCE_FILE, repo_target)
        print(f"[SYNC] Updated repo file: {repo_target}")
    else:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] AmrFlux data unchanged since last copy.")

    ts = last_timestamp(repo_target)
    msg = f"Update AmrFlux data and dashboard files (last record {ts})" if ts else "Update AmrFlux data and dashboard files"

    git_push_if_changed(
        repo_dir=GIT_REPO_DIR,
        message=msg,
        branch=GIT_BRANCH,
    )


def main():
    parser = argparse.ArgumentParser(description="Push LoggerNet AmrFlux data and dashboard files to GitHub.")
    parser.add_argument("--watch", action="store_true",
                        help=f"Keep running and check every {POLL_MINUTES} min")
    args = parser.parse_args()

    if not args.watch:
        sync_once()
        return

    print(f"=== Watching {SOURCE_FILE} every {POLL_MINUTES} min (Ctrl+C to stop) ===")
    while True:
        try:
            sync_once()
        except Exception as e:  # keep the watcher alive on transient errors (network, lock)
            print(f"[ERROR] {e}")
        time.sleep(POLL_MINUTES * 60)


if __name__ == "__main__":
    main()