#!/usr/bin/env bash
set -e

# Configure Go environment; GOROOT omitted to use system default
export GOPROXY=https://proxy.golang.org,direct
export GOSUMDB=sum.golang.org

# Run the main program
# Usage: ./run.sh [go run flags]
# Example: ./run.sh -lr=0.005 -epochs=20
cd "$(dirname "$0")"
go run load.go "$@"
