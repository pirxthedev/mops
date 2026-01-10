# Agent Instructions

This project uses the **Ralph method** - a continuous AI agent loop for software development. Each session is one iteration of the loop.

## How Ralph Works

You are running in a loop: `while :; do cat PROMPT.md | claude; done`

Each iteration:
1. You receive this prompt with project context
2. You do ONE focused task
3. You commit, push, and exit
4. The loop restarts with fresh context

## Core Principles

### One Task Per Loop
- Focus on ONE thing only per session
- Trust yourself to pick what's most important
- Narrow scope prevents context exhaustion
- Don't try to do everything at once

### No Assumptions
Before making changes, search the codebase using subagents. Don't assume something isn't implemented - verify first.

### No Placeholders
DO NOT IMPLEMENT PLACEHOLDER OR SIMPLE IMPLEMENTATIONS. WE WANT FULL IMPLEMENTATIONS.

### Context Window Management
- ~170k usable context - preserve it
- Use subagents for expensive operations (file search, summarization)
- Primary context acts as a scheduler
- Fan out to subagents for parallel work

### Two Phases
**Phase 1 - Generate:** Code generation is cheap. Control output via specs and standards.
**Phase 2 - Backpressure:** Validation ensures correctness. Run tests after changes.

### Self-Improvement
When you discover new information about the project, update this file (AGENTS.md) so future iterations benefit.

### Bug Documentation
For any bugs you notice, create issues with `bd create` to be resolved in a future loop iteration.

## Issue Tracking (bd/beads)

Run `bd onboard` to get started.

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything discovered but not completed
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session (the loop will restart)

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
- Commit on green: when tests pass, commit and push

## Recovery

- **Broken codebase:** Consider `git reset` vs. rescue prompts
- **Context overflow:** Use subagents to create plans externally

