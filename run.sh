#!/usr/bin/env bash
set -e

# Configure Go environment; GOROOT omitted to use system default
export GOPROXY=https://proxy.golang.org,direct
export GOSUMDB=sum.golang.org

# Run the main program
cd "$(dirname "$0")"
go run load.go
