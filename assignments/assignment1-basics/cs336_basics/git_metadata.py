"""Best-effort code identity and optional local evidence for dirty runs."""

import json
from pathlib import Path
import subprocess


def read_git_metadata(root, *, output=None):
    # None means unknown, not a clean checkout. Keep a known commit even if the
    # status check fails (for example, when iCloud files cannot be read).
    metadata = {"git_commit": None, "git_dirty": None, "git_error": None}

    def git(*args):
        result = subprocess.run(
            ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", *args],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        # Treat partial-read warnings as failure too: never label such a run clean.
        if result.returncode or result.stderr.strip():
            raise RuntimeError("Git could not reliably read the commit or working-tree status")
        return result.stdout

    try:
        metadata["git_commit"] = git("rev-parse", "--verify", "HEAD").strip()
        # Include staged, unstaged, and untracked files across the whole repo.
        # Ignored files (datasets, run outputs, local.toml) do not make it dirty.
        metadata["git_dirty"] = bool(git("status", "--porcelain=v1", "--untracked-files=normal").strip())
        if metadata["git_dirty"] and output is not None:
            # HEAD -> working tree includes staged AND unstaged tracked changes.
            # Binary patches are included; external diff programs are not invoked.
            # The top-level pathspec includes sibling assignments as well.
            diff = git("diff", "--binary", "--no-ext-diff", "--no-textconv", "--no-relative", "HEAD", "--", ":/")
            untracked = git("ls-files", "--others", "--exclude-standard", "--full-name", "-z", "--", ":/")
            # List names only, never copy untracked contents (which may be private
            # or huge). This evidence is NOT a complete source snapshot.
            with (Path(output) / "code.diff").open("x") as stream:
                stream.write(diff)
            with (Path(output) / "untracked_files.json").open("x") as stream:
                json.dump([name for name in untracked.split("\0") if name], stream, indent=2)
                stream.write("\n")
    except subprocess.TimeoutExpired:
        metadata["git_error"] = "Git metadata check timed out (5 seconds per command)"
    except (OSError, RuntimeError) as error:
        metadata["git_error"] = str(error)
    return metadata
