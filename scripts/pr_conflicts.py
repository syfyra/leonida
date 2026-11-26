"""Which open pull requests conflict with each other, and which merge without making another need a rebase.

    python -I scripts/pr_conflicts.py [PR ...] [--base REF] [--remote NAME] [--no-fetch] [--jobs N] [--json]

Merging one pull request makes others need a rebase, and nothing shows which ones: `needs rebase` is put on a pull
request only after `main` has moved under it. This asks git ahead of time, with no API call and no token. It takes
the open pull requests as `scripts/pr_overlap.py` finds and fetches them (the heads the remote has a
`refs/pull/N/merge` for, kept in `refs/pr-overlap/N`) and merges them pairwise in memory:

- each one with `--base` (default `origin/main`): a pull request that does not merge needs a rebase already;
- every two of the others that change a common file: if the two do not merge with each other, merging one makes
  the other need a rebase.

With no argument it prints the whole picture: how many open pull requests conflict with the base, how many of the
rest share no file with another or conflict with none (these merge in any order), the files with the most
conflicting pairs and the pull requests that conflict with the most others.

With pull request numbers, a merge round, it looks at those: which of them conflict with the base or with each
other, an order that keeps as many as possible mergeable (the one with the fewest conflicts in the round first,
the ones it conflicts with set aside, and so on), and for each how many other open pull requests its merge would
send to a rebase.

A pair is merged with `git merge-tree --stdin`, a few processes at a time, and the answer is kept per pair of
commits in `pr-conflicts.json` in the git directory, so a later run merges only the pairs with a new head. The
first run on a thousand open pull requests takes minutes. A conflict is git's: two pull requests that change the
same lines the same way still count, and two that merge cleanly can still break each other's tests.

Exit 0 when it ran, 2 on a usage or git error. Writes `refs/pr-overlap/*` (through pr_overlap) and the one cache
file. Standard library only; needs a git whose `merge-tree` has `--stdin`. No game files.
"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pr_overlap as overlap  # noqa: E402

CACHE_NAME = "pr-conflicts.json"
CACHE_VERSION = 1
SHOWN = 10


def merge_pairs(repo, pairs, jobs):
    """{(a, b): None when commits a and b merge, else the files in conflict}, for a list of commit pairs."""
    def run(chunk):
        if not chunk:
            return []
        r = overlap.git(repo, "merge-tree", "--write-tree", "--name-only", "--no-messages", "--stdin",
                        stdin="".join(f"{a} {b}\n" for a, b in chunk), check=False)
        if r.returncode != 0:
            raise overlap.GitError(f"git merge-tree --stdin: {r.stderr.strip() or r.returncode}"
                                   + (" (this git's merge-tree has no --stdin)" if r.returncode == 129 else ""))
        # One record per merge: clean ("1" or "0"), the merged tree, the files in conflict, an empty field.
        fields, at, found = r.stdout.split("\0"), 0, []
        for pair in chunk:
            try:
                end = fields.index("", at + 2)
            except ValueError:
                end = -1
            if end < 0 or fields[at] not in ("0", "1"):
                raise overlap.GitError("git merge-tree --stdin: output not understood")
            # A merge that is not clean always counts, even if git names no file for it.
            found.append((pair, None if fields[at] == "1" else fields[at + 2:end] or ["(no file named)"]))
            at = end + 1
        return found

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return dict(item for part in pool.map(run, [pairs[i::jobs] for i in range(jobs)]) for item in part)


class Cache:
    """Merge results per pair of commits, in a file in the git directory."""

    def __init__(self, repo):
        common = overlap.git(repo, "rev-parse", "--git-common-dir").stdout.strip()
        self.path = os.path.join(common if os.path.isabs(common) else os.path.join(repo, common), CACHE_NAME)
        self.pairs = {}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == CACHE_VERSION and isinstance(data.get("pairs"), dict):
                self.pairs = data["pairs"]
        except (OSError, ValueError, AttributeError):
            pass   # no cache yet, or one that cannot be read: merge everything again

    @staticmethod
    def key(a, b):
        return " ".join(sorted((a, b)))

    def get(self, a, b):
        """The files in conflict ([] when the two merge), or None when the pair is not known."""
        return self.pairs.get(self.key(a, b))

    def save(self, live):
        """Write the pairs whose two commits are in `live` (the base and the open heads), so the file does not
        grow with every head that was ever open, and a run for a few pull requests keeps what a full run found."""
        self.pairs = {key: files for key, files in sorted(self.pairs.items()) if set(key.split(" ")) <= live}
        folder = os.path.dirname(self.path)
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=folder, delete=False,
                                             prefix=CACHE_NAME + ".") as f:
                json.dump({"version": CACHE_VERSION, "pairs": self.pairs}, f, separators=(",", ":"))
            os.replace(f.name, self.path)
        except OSError as e:
            print(f"pr_conflicts: cache not written: {e.strerror or e}", file=sys.stderr)


def conflicts_of(repo, shas, wanted, cache, jobs):
    """{(a, b): files in conflict} for the pairs of pull request numbers in `wanted` that do not merge.
    Returns it with the number of pairs that had to be merged now."""
    missing = [(a, b) for a, b in wanted if cache.get(shas[a], shas[b]) is None]
    if missing:
        if len(missing) > 500:
            print(f"pr_conflicts: merging {len(missing)} pairs of pull requests, {jobs} at a time", file=sys.stderr)
        merged = merge_pairs(repo, [(shas[a], shas[b]) for a, b in missing], jobs)
        for (a, b), files in merged.items():
            cache.pairs[cache.key(a, b)] = files or []
    found = {(a, b): cache.get(shas[a], shas[b]) for a, b in wanted}
    return {pair: files for pair, files in found.items() if files}, len(missing)


def survey(repo, base, pulls, chosen, jobs, cache):
    """The conflict report for the open pull requests `pulls` ({number: head}); `chosen` narrows it to a round."""
    def files_of(item):
        return item[0], overlap.changed_files(repo, base, item[1])

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        files = dict(pool.map(files_of, pulls.items()))
    unrelated = sorted(n for n, changed in files.items() if changed is None)
    related = {n: sha for n, sha in pulls.items() if files[n] is not None}
    base_sha = overlap.commit(repo, base)
    # A pull request and the base are a pair like any other; the base's "number" is 0.
    shas = {0: base_sha, **related}
    stale, merged_now = conflicts_of(repo, shas, [(0, n) for n in related], cache, jobs)
    needs_rebase = {n: paths for (_, n), paths in stale.items()}
    mergeable = [n for n in related if n not in needs_rebase]

    by_file = defaultdict(list)
    for n in mergeable:
        for path in files[n]:
            by_file[path].append(n)
    wanted = set()
    for numbers in by_file.values():
        wanted.update(pair for pair in combinations(numbers, 2) if not chosen or set(pair) & set(chosen))
    conflicts, merged_pairs = conflicts_of(repo, shas, sorted(wanted), cache, jobs)
    cache.save(set(shas.values()))

    against = defaultdict(dict)
    for (a, b), paths in conflicts.items():
        against[a][b] = paths
        against[b][a] = paths
    shares = {n for numbers in by_file.values() if len(numbers) > 1 for n in numbers}
    return {"base": base, "open": len(pulls), "unrelated": unrelated, "needs_rebase": needs_rebase,
            "mergeable": mergeable, "alone": [n for n in mergeable if n not in shares],
            "free": [n for n in mergeable if n in shares and n not in against],
            "against": {n: dict(sorted(others.items())) for n, others in sorted(against.items())},
            "pairs": len(wanted), "merged_now": merged_now + merged_pairs,
            "by_file": {path: numbers for path, numbers in by_file.items() if len(numbers) > 1}}


def plan(numbers, against):
    """An order for a round: the pull request with the fewest conflicts among those left goes first, and the ones
    it conflicts with are set aside. Returns (the ones to merge, {set aside: the one it conflicts with})."""
    left, order, aside = set(numbers), [], {}
    while left:
        first = min(left, key=lambda n: (len(against.get(n, {}).keys() & left), n))
        order.append(first)
        left.discard(first)
        for other in sorted(against.get(first, {}).keys() & left):
            aside[other] = first
            left.discard(other)
    return order, aside


def numbered(numbers, limit=None):
    shown = [f"#{n}" for n in numbers[:limit]]
    return ", ".join(shown) + (f" and {len(numbers) - limit} more" if limit and len(numbers) > limit else "")


def print_overview(report):
    mergeable, tangled = report["mergeable"], sorted(report["against"])
    print(f"{report['open']} open pull request(s); {report['pairs']} pair(s) that change a common file were merged "
          f"with each other ({report['merged_now']} merge(s) run now, the rest from the last run)")
    if report["unrelated"]:
        print(f"  {len(report['unrelated'])} share no history with {report['base']}: "
              f"{numbered(report['unrelated'], SHOWN)}")
    print(f"  {len(report['needs_rebase'])} do not merge with {report['base']} and need a rebase")
    print(f"  {len(mergeable)} merge with {report['base']}:")
    print(f"    {len(report['alone'])} change no file another one changes")
    print(f"    {len(report['free'])} share a file with others and merge with all of them")
    print(f"    {len(tangled)} conflict with at least one other: merging one sends the other to a rebase")
    pairs = Counter()
    for a, others in report["against"].items():
        for b, paths in others.items():
            if a < b:
                pairs.update(paths)
    if pairs:
        print("\nFiles with the most conflicting pairs")
        for path, count in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:SHOWN]:
            print(f"  {count:5d} pair(s), {len(report['by_file'].get(path, [])):4d} pull request(s)  {path}")
        print("\nPull requests that conflict with the most others")
        for n in sorted(tangled, key=lambda n: (-len(report["against"][n]), n))[:SHOWN]:
            print(f"  #{n}: {len(report['against'][n])} ({numbered(list(report['against'][n]), 8)})")


def print_round(report, chosen):
    against = report["against"]
    stale = [n for n in chosen if n in report["needs_rebase"]]
    ready = [n for n in chosen if n in report["mergeable"]]
    order, aside = plan(ready, against)
    print(f"{len(chosen)} pull request(s) in the round, {report['open']} open")
    for n in stale:
        print(f"  #{n} does not merge with {report['base']}: {overlap.listing(report['needs_rebase'][n])}")
    for n in [n for n in chosen if n in report["unrelated"]]:
        print(f"  #{n} shares no history with {report['base']}")
    print(f"\nMerge, in this order or any other: {numbered(order) or 'none'}")
    for n in order:
        outside = [m for m in against.get(n, {}) if m not in chosen]
        if outside:
            print(f"  #{n} then sends {len(outside)} other open pull request(s) to a rebase: {numbered(outside, 8)}")
    if aside:
        print("\nSet aside, each conflicts with one of those")
        for n, first in sorted(aside.items()):
            print(f"  #{n} with #{first}: {overlap.listing(against[n][first])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("numbers", nargs="*", type=int, metavar="PR",
                    help="the pull requests of a merge round; with none, every open pull request")
    ap.add_argument("--base", default="origin/main", help="the branch pull requests merge into (default origin/main)")
    ap.add_argument("--remote", help="remote name or URL that has the pull requests (default: the remote of --base)")
    ap.add_argument("--repo", default=".", help="repository folder (default: the current one)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="do not contact the remote: use the heads the last run fetched")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4), help="git processes at once")
    ap.add_argument("--json", action="store_true", help="one JSON report instead of text")
    args = ap.parse_args(argv)
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    try:
        top = overlap.git(args.repo, "rev-parse", "--show-toplevel").stdout.strip()
        if not overlap.commit(top, args.base):
            raise overlap.GitError(f"--base {args.base} is not a commit here (git fetch first?)")
        if args.no_fetch:
            pulls = overlap.local_pull_requests(top)
            if not pulls:
                raise overlap.GitError("no pull request heads here yet: run once without --no-fetch")
        else:
            pulls = overlap.sync(top, args.remote or overlap.default_remote(top, args.base))[0]
        chosen = list(dict.fromkeys(args.numbers))
        closed = [n for n in chosen if n not in pulls]
        if closed:
            raise overlap.GitError(f"not an open pull request: {numbered(closed)}")
        report = survey(top, args.base, pulls, chosen, args.jobs, Cache(top))
    except overlap.GitError as e:
        print(f"pr_conflicts: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({**report, "round": chosen}, indent=1))
    elif chosen:
        print_round(report, chosen)
    else:
        print_overview(report)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
