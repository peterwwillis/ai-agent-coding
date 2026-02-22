#!/usr/bin/env bash
# lock_screen.sh - Adds a launch agent and script to lock MacOS screen on first login
set -u

SCRIPT="$( basename "${BASH_SOURCE[0]}" )"
SCRIPTDIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P )"
SCRIPTPATH="$SCRIPTDIR/$SCRIPT"

_add_login_startup_script () {
    mkdir -p ~/Library/LaunchAgents
    if [ ! -e  ~/Library/LaunchAgents/com.user.autolock.plist ] ; then

        cat <<EOF > ~/Library/LaunchAgents/com.user.autolock.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.autolock</string>
    <key>ProgramArguments</key>
    <array>
        <string>${SCRIPTPATH}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

    fi
}


_lock_screen () {

    # Try to lock screen with Python
    python3 -c "import ctypes; ctypes.CDLL('/System/Library/PrivateFrameworks/login.framework/login').SACLockScreenImmediate()"
    if [ $? -ne 0 ] ; then
        # Run keystroke to lock screen
        osascript -e 'tell application "System Events" to keystroke "q" using {control down, command down}'
    fi

    # Blank the display
    pmset displaysleepnow
}


if [ $# -eq 1 ] && [ "$1" = "install" ] ; then
    _add_login_startup_script
elif [ $# -eq 1 ] && [ "$1" = "lock" ] ; then
    _lock_screen
else
    _add_login_startup_script
    _lock_screen
fi
