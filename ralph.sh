#!/bin/bash
while :; do cat PROMPT.md | claude --dangerously-skip-permissions -p; done
