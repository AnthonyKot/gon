#!/bin/bash

# Set necessary Go environment variables
export GOROOT=/opt/homebrew/Cellar/go/1.24.0/libexec
export GOPROXY=https://proxy.golang.org,direct
export GOSUMDB=sum.golang.org

# Run the main program
cd "$(dirname "$0")"
go run load.go "$@"