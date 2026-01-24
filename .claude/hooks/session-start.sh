#!/bin/bash
# .claude/hooks/session-start.sh
# Installs bd (beads issue tracker) at the start of each Claude Code session

set -e  # Exit on error

echo "Setting up bd (beads issue tracker)..."

# Check if bd is already available
if command -v bd &> /dev/null; then
    echo "bd already installed: $(bd version)"
    # Still add go bin to PATH in case it was installed via go install
    export PATH="$PATH:$HOME/go/bin"
else
    echo "Installing bd..."

    # Try npm first (fastest when it works)
    if npm install -g @beads/bd --quiet 2>/dev/null && command -v bd &> /dev/null; then
        echo "Installed via npm"
    # Fall back to go install (works when npm postinstall fails)
    elif command -v go &> /dev/null; then
        echo "npm install failed, trying go install..."
        if go install github.com/steveyegge/beads/cmd/bd@latest 2>/dev/null; then
            export PATH="$PATH:$HOME/go/bin"
            echo "Installed via go install"
        else
            echo "go install failed (network issues?)"
            echo "You can retry manually: go install github.com/steveyegge/beads/cmd/bd@latest"
            echo "Then: export PATH=\$PATH:\$HOME/go/bin"
            exit 1
        fi
    else
        echo "Installation failed - neither npm nor go available"
        exit 1
    fi
fi

# Verify installation
if bd version &> /dev/null; then
    echo "bd $(bd version) ready"
else
    echo "bd installation verification failed"
    exit 1
fi

# Initialize if needed (but .beads should already exist in this project)
if [ ! -d .beads ]; then
    echo "Initializing bd in project..."
    bd init --quiet
else
    echo ".beads directory found"
fi

# Show brief status
echo ""
echo "Issue summary:"
bd ready --limit 5 2>/dev/null || echo "(no issues ready)"
echo ""
echo "Use 'bd --help' for commands."
