#!/bin/bash
while :; do cat "$1" | claude --dangerously-skip-permissions -p; done
