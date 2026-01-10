# PLAN.md

0a. Study docs/plans/* and specs/* for project specifications.

0b. Source code is in mops-core/ and mops-python/.

0c. Run `bd ready` to see available work.

1. Implement from ready queue using `bd ready --limit 10`. Before changes, search codebase using subagents (don't assume not implemented). Up to 500 parallel subagents for search, only 1 for build/tests.

2. After implementing, run tests for that unit. If functionality missing, add it per specs. Think hard.

3. When you discover bugs or issues: `bd create "Bug: X" --type=bug --priority=1` immediately using a subagent.

4. When tests pass: `git add -A && git commit -m "description" && bd sync && git push`

999. When authoring docs, capture why tests and implementation choices matter.

9999. Single source of truth. No migrations/adapters. Fix unrelated failing tests as part of your change.

99999. When all tests pass, create git tag. Start at 0.0.0, increment patch.

999999. Add logging if needed for debugging.

9999999. Close issues immediately when done: `bd close <id> --reason="description"`

99999999. Update AGENTS.md with build/test learnings using a subagent. Keep brief. NO STATUS REPORTS.

999999999. DO NOT IMPLEMENT PLACEHOLDERS. FULL IMPLEMENTATIONS ONLY.

9999999999. Before ending: `bd sync && git push`. Work is not done until pushed.
