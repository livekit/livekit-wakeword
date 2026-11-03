#!/bin/sh
# Set the release version everywhere it is written down, then sync uv.lock:
#   pyproject.toml                                    the livekit-wakeword Python package
#   client/esp32/livekit_wakeword/idf_component.yml   the ESP32 component
# (Swift Package Manager versions come from the git tag.) Commit the result in a
# "Release vX.Y.Z" PR; the publish workflow checks that the pushed tag matches.
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

edit() {  # edit <file> <sed expression>
    sed "$2" "$1" > "$1.tmp"
    mv "$1.tmp" "$1"
    echo "updated $1"
}

edit "$REPO_ROOT/pyproject.toml" "s/^version = \".*\"/version = \"$VERSION\"/"
edit "$REPO_ROOT/client/esp32/livekit_wakeword/idf_component.yml" \
    "s/^version: \".*\"/version: \"$VERSION\"/"

(cd "$REPO_ROOT" && uv lock)
echo "updated $REPO_ROOT/uv.lock"
