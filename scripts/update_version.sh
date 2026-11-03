#!/bin/sh
# Set the package version in pyproject.toml and livekit.wakeword.__version__, then sync
# uv.lock. Commit all three together in a "Release vX.Y.Z" PR; the publish workflow checks
# that the pushed tag matches this version.
# Usage: scripts/update_version.sh <version>
set -eu

if [ $# -ne 1 ]; then
    echo "usage: $0 <version>" >&2
    exit 1
fi

VERSION=$1
if ! echo "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?$'; then
    echo "error: '$VERSION' is not a semver (expected X.Y.Z or X.Y.Z-rc.N)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

file="$REPO_ROOT/pyproject.toml"
sed "s/^version = \".*\"/version = \"$VERSION\"/" "$file" > "$file.tmp"
mv "$file.tmp" "$file"
echo "updated $file"

file="$REPO_ROOT/src/livekit/wakeword/__init__.py"
sed "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" "$file" > "$file.tmp"
mv "$file.tmp" "$file"
echo "updated $file"

(cd "$REPO_ROOT" && uv lock)
echo "updated $REPO_ROOT/uv.lock"
