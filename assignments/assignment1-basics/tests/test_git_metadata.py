"""Standard-library-only tests so metadata checks do not require ML packages."""

from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cs336_basics.git_metadata import read_git_metadata


class GitMetadataTests(unittest.TestCase):
    def test_clean_modified_staged_untracked_and_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
            git("init")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.invalid")
            (root / ".gitignore").write_text("ignored\n")
            (root / "code.txt").write_text("original")
            git("add", ".gitignore", "code.txt")
            git("-c", "commit.gpgsign=false", "commit", "-m", "fixture")
            commit = git("rev-parse", "HEAD").stdout.strip()
            (root / "ignored").write_text("not code")
            self.assertEqual(read_git_metadata(root), {
                "git_commit": commit, "git_dirty": False, "git_error": None,
            })
            (root / "code.txt").write_text("modified")
            self.assertTrue(read_git_metadata(root)["git_dirty"])
            git("add", "code.txt")
            self.assertTrue(read_git_metadata(root)["git_dirty"])
            git("-c", "commit.gpgsign=false", "commit", "-m", "update")
            (root / "new.txt").write_text("untracked")
            self.assertTrue(read_git_metadata(root)["git_dirty"])
            (root / "code.txt").write_text("staged change")
            git("add", "code.txt")
            (root / "code.txt").write_text("staged change plus unstaged change\n")
            # Run from a subdirectory to verify the whole repository is covered.
            (root / "subdir").mkdir()
            with tempfile.TemporaryDirectory() as artifacts:
                record = read_git_metadata(root / "subdir", output=artifacts)
                self.assertIsNone(record["git_error"])
                diff = (Path(artifacts) / "code.diff").read_text()
                self.assertIn("+staged change plus unstaged change", diff)
                self.assertNotIn("new.txt", diff)
                self.assertEqual(json.loads((Path(artifacts) / "untracked_files.json").read_text()), ["new.txt"])

    def test_missing_git(self):
        with patch("cs336_basics.git_metadata.subprocess.run", side_effect=FileNotFoundError("git missing")):
            record = read_git_metadata(".")
        self.assertIsNone(record["git_commit"])
        self.assertIsNone(record["git_dirty"])
        self.assertTrue(record["git_error"])

    def test_timeout_preserves_commit_but_not_clean_status(self):
        replies = [subprocess.CompletedProcess([], 0, "abc123\n", ""),
                   subprocess.TimeoutExpired("git status", 5)]
        with patch("cs336_basics.git_metadata.subprocess.run", side_effect=replies) as command:
            record = read_git_metadata(".")
        self.assertEqual(record["git_commit"], "abc123")
        self.assertIsNone(record["git_dirty"])
        self.assertIn("timed out", record["git_error"])
        self.assertEqual(command.call_args.kwargs["timeout"], 5)
        self.assertIn("--no-optional-locks", command.call_args.args[0])

    def test_unreadable_head_and_partial_status(self):
        for replies in (
            [subprocess.CompletedProcess([], 128, "", "unreadable HEAD")],
            [subprocess.CompletedProcess([], 0, "abc123\n", ""),
             subprocess.CompletedProcess([], 0, "", "short read while indexing")],
        ):
            with self.subTest(replies=replies):
                with patch("cs336_basics.git_metadata.subprocess.run", side_effect=replies):
                    record = read_git_metadata(".")
                self.assertIsNone(record["git_dirty"])
                self.assertTrue(record["git_error"])


if __name__ == "__main__":
    unittest.main()
