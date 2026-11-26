"""scripts/pr_conflicts.py on throwaway git repositories (no network, no game data).

    python -I tests/test_pr_conflicts.py

The repositories are the ones tests/test_pr_overlap.py builds: an "upstream" with pull request refs laid out as
GitHub does, and a clone of it where the script runs.
"""
import contextlib
import io
import json
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import pr_conflicts  # noqa: E402
from test_pr_overlap import PrOverlapTest, page  # noqa: E402

INDEX = "viewer/index.html"


class ConflictsCase(PrOverlapTest):
    def run_tool(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = pr_conflicts.main(["--repo", self.clone, *args])
        return code, out.getvalue(), err.getvalue()

    def report(self, *args):
        code, out, err = self.run_tool("--json", *args)
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def pulls(self):
        """1, 2 and 5 change the same line; 3 changes the same file elsewhere; 4 a file of its own; 6 a file that
        main changes afterwards; 7 is closed; 8 has no history in common."""
        shas = {1: self.pull(1, {INDEX: page(line5="one")}),
                2: self.pull(2, {INDEX: page(line5="two")}),
                3: self.pull(3, {INDEX: page(line11="three")}),
                4: self.pull(4, {"tools/a.py": "a = 4\n"}),
                5: self.pull(5, {INDEX: page(line5="five"), "viewer/studio.html": page(line3="five")}),
                6: self.pull(6, {"docs/world.md": "# World\n\nSix.\n"}),
                7: self.pull(7, {INDEX: page(line5="closed")}, is_open=False)}
        self.git(self.upstream, "checkout", "-q", "--orphan", "island")
        shas[8] = self.commit(self.upstream, "another history")
        for kind in ("head", "merge"):
            self.git(self.upstream, "update-ref", f"refs/pull/8/{kind}", shas[8])
        self.git(self.upstream, "checkout", "-q", "main")
        self.write(self.upstream, "docs/world.md", "# World\n\nMain moved on.\n")
        self.commit(self.upstream, "main changes the file of pull request 6")
        self.git(self.clone, "fetch", "-q", "origin")
        return shas

    def cache(self):
        with open(os.path.join(self.clone, ".git", "pr-conflicts.json"), encoding="utf-8") as f:
            return json.load(f)


class OverviewTest(ConflictsCase):
    def test_the_whole_picture(self):
        self.pulls()
        report = self.report()
        self.assertEqual((report["open"], report["unrelated"], report["round"]), (7, [8], []))
        self.assertEqual(report["needs_rebase"], {"6": ["docs/world.md"]})
        self.assertEqual((report["mergeable"], report["alone"], report["free"]), ([1, 2, 3, 4, 5], [4], [3]))
        # 1, 2 and 5 conflict with each other; 3 shares their file and merges with all three.
        self.assertEqual(report["against"], {"1": {"2": [INDEX], "5": [INDEX]}, "2": {"1": [INDEX], "5": [INDEX]},
                                             "5": {"1": [INDEX], "2": [INDEX]}})
        self.assertEqual(report["by_file"], {INDEX: [1, 2, 3, 5]})
        # Six pull requests against main, and the six pairs among the four that change the page.
        self.assertEqual((report["pairs"], report["merged_now"]), (6, 12))

    def test_text_overview(self):
        self.pulls()
        code, out, err = self.run_tool()
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out, (
            "7 open pull request(s); 6 pair(s) that change a common file were merged with each other "
            "(12 merge(s) run now, the rest from the last run)\n"
            "  1 share no history with origin/main: #8\n"
            "  1 do not merge with origin/main and need a rebase\n"
            "  5 merge with origin/main:\n"
            "    1 change no file another one changes\n"
            "    1 share a file with others and merge with all of them\n"
            "    3 conflict with at least one other: merging one sends the other to a rebase\n"
            "\nFiles with the most conflicting pairs\n"
            "      3 pair(s),    4 pull request(s)  viewer/index.html\n"
            "\nPull requests that conflict with the most others\n"
            "  #1: 2 (#2, #5)\n  #2: 2 (#1, #5)\n  #5: 2 (#1, #2)\n"))

    def test_files_are_listed_by_how_many_pairs_conflict_in_them(self):
        for n in (1, 2, 3):
            self.pull(n, {INDEX: page(line5=f"pull request {n}")})
        for n in (4, 5):
            self.pull(n, {"tools/a.py": f"a = {n}\n"})
        out = self.run_tool()[1]
        # Three pairs in the page, one in the tool: the page first, although the tool comes first by name.
        self.assertIn("\nFiles with the most conflicting pairs\n"
                      "      3 pair(s),    3 pull request(s)  viewer/index.html\n"
                      "      1 pair(s),    2 pull request(s)  tools/a.py\n", out)
        self.assertIn("\nPull requests that conflict with the most others\n"
                      "  #1: 2 (#2, #3)\n  #2: 2 (#1, #3)\n  #3: 2 (#1, #2)\n  #4: 1 (#5)\n  #5: 1 (#4)\n", out)

    def test_nothing_in_conflict_prints_no_tables(self):
        self.pull(1, {INDEX: page(line5="one")})
        self.pull(2, {"tools/a.py": "a = 2\n"})
        out = self.run_tool()[1]
        self.assertIn("    2 change no file another one changes\n", out)
        self.assertIn("    0 conflict with at least one other", out)
        self.assertNotIn("Files with the most", out)
        self.assertNotIn("share no history", out)


class CacheTest(ConflictsCase):
    def test_a_later_run_merges_only_what_has_a_new_head(self):
        shas = self.pulls()
        self.assertEqual(self.report()["merged_now"], 12)
        cache = self.cache()
        self.assertEqual((cache["version"], len(cache["pairs"])), (1, 12))
        self.assertEqual(cache["pairs"][" ".join(sorted((shas[1], shas[2])))], [INDEX])
        self.assertEqual(cache["pairs"][" ".join(sorted((shas[1], shas[3])))], [])
        again = self.report("--no-fetch")
        self.assertEqual((again["merged_now"], again["against"]["1"]), (0, {"2": [INDEX], "5": [INDEX]}))
        # Pull request 2 moves to another line: one merge with main and three with 1, 3 and 5.
        self.pull(2, {INDEX: page(line8="two, elsewhere")})
        report = self.report()
        self.assertEqual((report["merged_now"], report["free"]), (4, [2, 3]))
        self.assertEqual(report["against"], {"1": {"5": [INDEX]}, "5": {"1": [INDEX]}})
        # The pairs of the head that is gone are dropped; the count stays at twelve.
        self.assertEqual(len(self.cache()["pairs"]), 12)
        self.assertFalse(any(shas[2] in key for key in self.cache()["pairs"]))

    def test_a_round_keeps_what_a_full_run_found(self):
        self.pulls()
        self.report()
        self.assertEqual(self.report("4")["merged_now"], 0)
        self.assertEqual(len(self.cache()["pairs"]), 12)
        self.assertEqual(self.report()["merged_now"], 0)

    def test_a_cache_that_cannot_be_read_is_rebuilt(self):
        self.pulls()
        self.report()
        path = os.path.join(self.clone, ".git", "pr-conflicts.json")
        # A file of another version is not trusted, whatever it says: here, that every pair merges.
        other_version = json.dumps({"version": 0, "pairs": {key: [] for key in self.cache()["pairs"]}})
        for text in ("{not json", other_version, '["a list"]'):
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            report = self.report("--no-fetch")
            self.assertEqual((report["merged_now"], sorted(report["against"])), (12, ["1", "2", "5"]), text)
        self.assertEqual(self.cache()["version"], 1)
        self.assertEqual(pr_conflicts.Cache.key("b", "a"), pr_conflicts.Cache.key("a", "b"))

    def test_a_round_merges_only_the_pairs_of_its_pull_requests(self):
        self.pulls()
        report = self.report("4")
        self.assertEqual((report["pairs"], report["merged_now"], report["round"]), (0, 6, [4]))
        report = self.report("3")
        self.assertEqual((report["pairs"], report["merged_now"]), (3, 3))


class RoundTest(ConflictsCase):
    def test_a_round_is_ordered_and_says_what_each_merge_costs(self):
        self.pulls()
        code, out, err = self.run_tool("1", "2", "3", "4", "6", "3")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out, (
            "5 pull request(s) in the round, 7 open\n"
            "  #6 does not merge with origin/main: docs/world.md\n"
            "\nMerge, in this order or any other: #3, #4, #1\n"
            "  #1 then sends 1 other open pull request(s) to a rebase: #5\n"
            "\nSet aside, each conflicts with one of those\n"
            "  #2 with #1: viewer/index.html\n"))
        self.assertEqual(self.report("1", "2", "3", "4", "6", "3")["round"], [1, 2, 3, 4, 6])

    def test_a_round_without_conflicts_and_one_with_nothing_to_merge(self):
        self.pulls()
        out = self.run_tool("3", "4", "8")[1]
        self.assertEqual(out, "3 pull request(s) in the round, 7 open\n  #8 shares no history with origin/main\n"
                              "\nMerge, in this order or any other: #3, #4\n")
        self.assertIn("\nMerge, in this order or any other: none\n", self.run_tool("6")[1])

    def test_the_fewest_conflicts_go_first(self):
        # A chain 1 - 2 - 3: taking 2 first would set both others aside.
        chain = {1: {2: ["f"]}, 2: {1: ["f"], 3: ["f"]}, 3: {2: ["f"]}}
        self.assertEqual(pr_conflicts.plan([1, 2, 3], chain), ([1, 3], {2: 1}))
        self.assertEqual(pr_conflicts.plan([3, 2], chain), ([2], {3: 2}))
        self.assertEqual(pr_conflicts.plan([5, 4], {}), ([4, 5], {}))
        # A neighbour outside the round does not count against a pull request.
        self.assertEqual(pr_conflicts.plan([2, 3, 9], {**chain, 9: {}}), ([9, 2], {3: 2}))


class ErrorsTest(ConflictsCase):
    def test_usage_and_git_errors_exit_2_with_one_line(self):
        self.pulls()
        code, out, err = self.run_tool("--no-fetch")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("pr_conflicts: no pull request heads here yet", err)
        for args, message in ((["1", "7", "99"], "pr_conflicts: not an open pull request: #7, #99\n"),
                              (["--base", "origin/nowhere"], "pr_conflicts: --base origin/nowhere is not a commit here"),
                              (["--remote", "nowhere"], "pr_conflicts: git ls-remote nowhere")):
            code, out, err = self.run_tool(*args)
            self.assertEqual((code, out), (2, ""), args)
            self.assertTrue(err.startswith(message), err)
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stderr(io.StringIO()):
            pr_conflicts.main(["--repo", self.clone, "--jobs", "0"])
        self.assertEqual(stop.exception.code, 2)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            pr_conflicts.main(["--repo", self.clone, "abc"])

    def test_one_job_gives_the_same_answer(self):
        self.pulls()
        self.assertEqual(self.report("--jobs", "1")["against"], self.report("--jobs", "3")["against"])

    def test_merges_are_read_from_gits_records(self):
        shas = self.pulls()
        self.git(self.clone, "fetch", "-q", "origin", "refs/pull/*/head:refs/pr-overlap/*")
        pairs = [(shas[1], shas[2]), (shas[1], shas[3]), (shas[4], shas[5])]
        self.assertEqual(pr_conflicts.merge_pairs(self.clone, pairs, 2),
                         {pairs[0]: [INDEX], pairs[1]: None, pairs[2]: None})
        self.assertEqual(pr_conflicts.merge_pairs(self.clone, [], 4), {})

        def answer(code, out):
            return mock.patch.object(pr_conflicts.overlap, "git", return_value=types.SimpleNamespace(
                returncode=code, stdout=out, stderr="usage: git merge-tree"))

        with answer(129, ""), self.assertRaisesRegex(pr_conflicts.overlap.GitError, "has no --stdin"):
            pr_conflicts.merge_pairs(self.clone, pairs[:1], 1)
        with answer(128, ""), self.assertRaisesRegex(pr_conflicts.overlap.GitError, r"usage: git merge-tree$"):
            pr_conflicts.merge_pairs(self.clone, pairs[:1], 1)
        for broken in ("", "2\0tree\0\0", "0\0tree\0a.txt"):
            with answer(0, broken), self.assertRaisesRegex(pr_conflicts.overlap.GitError, "not understood"):
                pr_conflicts.merge_pairs(self.clone, pairs[:1], 1)
        # A merge git calls unclean without naming a file is still a conflict.
        with answer(0, "0\0tree\0\0"):
            self.assertEqual(pr_conflicts.merge_pairs(self.clone, pairs[:1], 1), {pairs[0]: ["(no file named)"]})


if __name__ == "__main__":
    unittest.main()
