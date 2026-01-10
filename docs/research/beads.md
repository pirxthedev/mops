# Beads: Git-Backed Issue Tracker for AI Agents

> Source: [github.com/steveyegge/beads](https://github.com/steveyegge/beads) by Steve Yegge (2025)

## Overview

Beads (bd) is a git-backed issue tracker designed specifically for AI coding agents. It replaces markdown-based planning files (like fix_plan.md and IMPLEMENTATION_PLAN.md) with a structured, dependency-aware task graph that persists across agent sessions.

## Why Beads Over Markdown Plans

| Markdown Plans (Ralph) | Beads |
|------------------------|-------|
| Single file, linear list | Dependency graph with blocking relationships |
| Manual priority ordering | Programmatic priority levels (P0-P4) |
| Text parsing required | JSON output for machine consumption |
| No state persistence | SQLite cache + JSONL for git sync |
| Regenerate when off-track | Query for ready tasks, update in-place |
| Context window overhead | ~1-2k token injection via `bd prime` |

## Core Architecture

### Three-Layer Design

1. **CLI Layer** - Cobra commands with `--json` for programmatic use
2. **SQLite Database** - Local cache (gitignored) for fast queries
3. **JSONL Files** - Git-tracked source of truth in `.beads/`

### Storage Structure

```
.beads/
├── beads.db          # SQLite (gitignored)
├── issues.jsonl      # Source of truth (tracked)
├── bd.sock           # Daemon socket
├── config.yaml       # Project config
└── export_hashes.db  # Sync tracking
```

### Sync Model

- Writes go to SQLite immediately
- 5-second debounce batches changes
- Incremental export to JSONL
- Git hooks can auto-commit changes
- `bd sync` flushes, commits, and pushes

## Replacing fix_plan.md

The Ralph technique uses `fix_plan.md` for:
- Prioritized list of remaining work
- Bug documentation discovered during development
- Regenerated when agent goes off-track

### Beads Equivalent

```bash
# Create tasks (replaces adding to fix_plan.md)
bd create "Fix auth token refresh" -t bug -p 1

# Get next work item (replaces parsing fix_plan.md)
bd ready --limit 1 --json

# Mark complete (replaces deleting from fix_plan.md)
bd close <id> --reason "Fixed in commit abc123"

# Document blockers (replaces manual notes)
bd update <id> --status blocked
bd note <id> "Waiting on API changes"
```

### Key Advantages

1. **Query by status**: `bd list --status open --priority 0`
2. **Dependency tracking**: Tasks with blockers don't appear in `bd ready`
3. **Atomic updates**: No file parsing/rewriting race conditions
4. **Audit trail**: Full history of task changes

## Replacing IMPLEMENTATION_PLAN.md

Implementation plans typically contain:
- Ordered list of features/tasks
- Dependencies between tasks
- Phase/milestone groupings

### Beads Equivalent: Molecules

Molecules are epics (parent issues with children) that define workflow structure:

```bash
# Create epic (the plan)
bd create "User Authentication System" -t epic -p 0

# Create subtasks with dependencies
bd create "Add JWT library" -t task -p 1 --parent <epic-id>
bd create "Implement login endpoint" -t task -p 1 --parent <epic-id>
bd create "Add protected routes" -t task -p 2 --parent <epic-id>

# Establish dependencies
bd dep add <routes-id> <login-id>  # routes depends on login
bd dep add <login-id> <jwt-id>    # login depends on jwt
```

### Dependency Tree Visualization

```bash
bd dep tree <epic-id>
```

Shows blocked/ready status for entire plan hierarchy.

## Essential Commands

### Discovery

| Command | Purpose |
|---------|---------|
| `bd ready` | Unblocked tasks ready for work |
| `bd ready --limit 1 --json` | Next task for automated loop |
| `bd stale --days 7` | Neglected tasks |
| `bd list --status in_progress` | Currently active work |

### Task Management

| Command | Purpose |
|---------|---------|
| `bd create "Title" -t type -p priority` | Create task |
| `bd update <id> --status in_progress` | Start work |
| `bd close <id> --reason "done"` | Complete task |
| `bd block <id> --reason "needs X"` | Mark blocked |
| `bd note <id> "context"` | Add notes |

### Dependencies

| Command | Purpose |
|---------|---------|
| `bd dep add <child> <parent>` | Create dependency |
| `bd dep tree <id>` | Visualize hierarchy |
| `bd dep remove <child> <parent>` | Remove dependency |

### Synchronization

| Command | Purpose |
|---------|---------|
| `bd sync` | Flush, commit, push |
| `bd import -i file.jsonl` | Import issues |
| `bd setup claude` | Install Claude hooks |

## Issue Properties

### Types
- `bug` - Defects
- `feature` - New functionality
- `task` - General work
- `epic` - Parent container
- `chore` - Maintenance

### Priorities
- `0` - Critical (P0)
- `1` - High
- `2` - Medium
- `3` - Low
- `4` - Backlog

### Statuses
- `open` - Not started
- `in_progress` - Active work
- `blocked` - Waiting on dependency
- `deferred` - Postponed
- `closed` - Complete

## Agent Loop Integration

### Basic Ralph Loop with Beads

```bash
while :; do
  # Get next task
  TASK=$(bd ready --limit 1 --json)

  if [ -z "$TASK" ]; then
    echo "All tasks complete"
    break
  fi

  # Inject context and run agent
  bd prime | cat - PROMPT.md | claude-code

  # Sync changes
  bd sync
done
```

### In-Agent Workflow

From CLAUDE.md, the recommended loop:

1. `bd ready --limit 1 --json` - Get next task
2. If empty - Output completion signal
3. Create checkpoint before work
4. Do the task
5. `bd close <id> --reason "done"` - Mark complete
6. Repeat

### Error Handling

- Task fails, environment OK: `bd note <id> "error"`, retry
- Stuck after 3 attempts: `bd block <id> "reason"`, next task
- Discover subtasks: `bd create "subtask" -t task -p 2 --parent <id>`

## Molecules for Complex Workflows

### Template-Based (Protos)

```bash
# Create template
bd create "Standard Feature Template" -t epic --label template

# Pour instance
bd mol pour <proto-id> --var name="Auth Feature"
```

### Ephemeral Work (Wisps)

```bash
# Create temporary structure (not synced)
bd mol wisp <proto-id> --var name="Quick Fix"

# Complete and discard
bd mol burn <wisp-id>

# Or compress to permanent record
bd mol squash <wisp-id> --summary "Fixed X, Y, Z"
```

### Phase Model

| Phase | Type | Synced | Use Case |
|-------|------|--------|----------|
| Solid | Proto | Yes | Reusable templates |
| Liquid | Mol | Yes | Active projects |
| Vapor | Wisp | No | One-off work |

## Claude Integration

### Setup

```bash
bd setup claude           # Install hooks
bd setup claude --check   # Verify setup
bd setup claude --stealth # Flush-only mode
```

### Hooks Installed

1. **SessionStart** - Runs `bd prime` to inject context
2. **PreCompact** - Preserves workflow before summarization

### Context Injection

`bd prime` outputs ~1-2k tokens of:
- Current ready tasks
- Active in-progress work
- Workflow instructions

## Stealth Mode

For contributing to repos without adding `.beads/`:

```bash
bd init --stealth
```

Tasks track locally only, enabling personal task management on shared projects.

## Key Differences from Markdown Plans

1. **Structured queries** vs text parsing
2. **Dependency blocking** prevents premature work
3. **Atomic operations** vs file rewrites
4. **Git-native** with merge-friendly JSONL
5. **Hash-based IDs** prevent collision in parallel work
6. **Audit trail** tracks all changes
7. **Token-efficient** context injection

## Migration from fix_plan.md

1. Install: `npm i -g @beads/bd` or `brew install beads`
2. Initialize: `bd init`
3. Import existing tasks: Parse markdown, create via `bd create`
4. Remove fix_plan.md from loop
5. Use `bd ready --limit 1 --json` for task selection
6. Add `bd sync` at session end
