# Ralph + Beads Loop

You are running in a Ralph loop inside a Sprites VM. Use beads for task state, checkpoints for safety.

## The Loop

1. `bd ready --limit 1 --json` → get next task
1. If empty → output `<promise>COMPLETE</promise>` and stop
1. `sprite-env checkpoints create` → save state before working
1. Do the task
1. `bd close <id> --reason "done"` → mark complete
1. Repeat

## If Things Break

- Task fails, environment OK → `bd note <id> "what went wrong"`, retry
- Task corrupts environment → `sprite-env checkpoints restore <previous>`, retry
- Stuck after 3 attempts → `bd block <id> "reason"`, move to next task

## Discovering Work

If you find subtasks: `bd create "subtask title" -t task -p 2`
