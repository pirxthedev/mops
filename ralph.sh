#!/bin/bash
while true; do
    cat PROMPT.md | claude -p \
        --dangerously-skip-permissions \
        --output-format=stream-json \
        --model opus \
        --verbose \
        | bunx repomirror visualize
    git push origin main
    echo -e "\n\n========================LOOP=========================\n\n"
done
