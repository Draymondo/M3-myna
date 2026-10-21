#!/usr/bin/env bash
# Run a command in a session isolated from the host desktop's: a private
# session bus and a private, short XDG_RUNTIME_DIR.
#
# Usage: dev/nested-session.sh COMMAND [ARGS...]
#   e.g. dev/nested-session.sh gnome-shell --headless --wayland --no-x11 \
#            --virtual-monitor 1600x1000
#
# Every headless or nested GNOME Shell, and anything else that may start an
# accessibility bus, goes through this. A private bus alone is not enough:
# whatever starts on it D-Bus-activates its own at-spi-bus-launcher, which
# names its socket $XDG_RUNTIME_DIR/at-spi/bus$DISPLAY and unlinks whatever is
# there before binding (at-spi2-core 2.62). Sharing the host's runtime dir, it
# replaces the desktop's live a11y socket, and once it exits that path is dead:
# every new AT-SPI client on the desktop, Orca included, then fails to connect
# until the user logs out. The private runtime dir also keeps the nested
# Shell's wayland-N socket off the host's.
#
# The dir sits under /tmp, not the caller's TMPDIR: an AF_UNIX path must fit
# 108 bytes and the launcher silently runs without a socket past 100.
#
#   NESTED_SHARE_AUDIO=1   link the host's PipeWire and Pulse sockets in, for
#                          commands that need real audio
#   NESTED_KEEP=1          leave the runtime dir behind (its path is printed)
set -euo pipefail

[ $# -gt 0 ] || { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//' >&2; exit 2; }

HOST_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
RUNTIME_DIR=$(mktemp -d /tmp/myna-rt.XXXXXX)
chmod 700 "$RUNTIME_DIR"
echo "nested-session: XDG_RUNTIME_DIR=$RUNTIME_DIR" >&2

cleanup() {
    [ "${NESTED_KEEP:-}" = 1 ] || rm -rf "$RUNTIME_DIR"
}
trap cleanup EXIT

if [ "${NESTED_SHARE_AUDIO:-}" = 1 ]; then
    for sock in pipewire-0 pipewire-0-manager; do
        if [ -S "$HOST_RUNTIME_DIR/$sock" ]; then
            ln -s "$HOST_RUNTIME_DIR/$sock" "$RUNTIME_DIR/$sock"
        fi
    done
    if [ -S "$HOST_RUNTIME_DIR/pulse/native" ]; then
        mkdir -p "$RUNTIME_DIR/pulse"
        ln -s "$HOST_RUNTIME_DIR/pulse/native" "$RUNTIME_DIR/pulse/native"
    fi
fi

# The host's display and bus addresses must not leak in: the nested command
# would talk to the real compositor, or reach the real session bus directly.
env -u DISPLAY -u WAYLAND_DISPLAY -u DBUS_SESSION_BUS_ADDRESS -u AT_SPI_BUS_ADDRESS \
    XDG_RUNTIME_DIR="$RUNTIME_DIR" \
    dbus-run-session -- "$@"
