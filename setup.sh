#!/usr/bin/env bash
# Install AgentFabric + all dependencies in one command.
# Run from the AgentFabric/ directory:
#   bash setup.sh

set -e

echo "Installing AgentFabric with all dependencies..."
pip install -e ".[all]"

echo ""
echo "Done! You can now import agentfabric in Python or Jupyter."
