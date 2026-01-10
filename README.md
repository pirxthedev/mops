# mops

Research repository for AI agent workflow techniques and tools.

## Development Setup

### Prerequisites

- Node.js (for beads installation)

### Install Beads

This project uses [Beads](https://github.com/steveyegge/beads) (bd) for git-backed issue tracking designed for AI coding agents.

```bash
npm install -g @beads/bd
```

### Initialize (already done for this repo)

If cloning fresh:

```bash
bd init
bd hooks install
bd setup claude  # For Claude Code integration
```

## Using Beads

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd create "Title" -t task -p 1  # Create a task
bd update <id> --status in_progress  # Start work
bd close <id>         # Complete work
bd sync               # Sync with git
```

See [AGENTS.md](./AGENTS.md) for agent workflow instructions.

## Research

- [docs/research/beads.md](./docs/research/beads.md) - Beads issue tracker notes
- [docs/research/ralph.md](./docs/research/ralph.md) - Ralph technique notes
