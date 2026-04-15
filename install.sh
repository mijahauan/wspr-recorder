#!/bin/bash
#
# wspr-recorder installation/upgrade script
#
# Installs or upgrades wspr-recorder as a systemd service with:
# - Virtual environment in /opt/wspr-recorder
# - Configuration in /etc/wspr-recorder
# - Runtime files in /run/wspr-recorder (tmpfs)
# - WAV output in /dev/shm/wspr-recorder (tmpfs)
# - Logs via journald
#
# The script is idempotent - running it multiple times will upgrade
# the installation, including all dependencies (e.g., ka9q-python).
#
# Usage:
#   sudo ./install.sh              # Install or upgrade
#   sudo ./install.sh --uninstall  # Remove installation
#

set -e

# Configuration
INSTALL_DIR="/opt/wspr-recorder"
CONFIG_DIR="/etc/wspr-recorder"
RUN_DIR="/run/wspr-recorder"
OUTPUT_DIR="/dev/shm/wspr-recorder"
LOG_DIR="/var/log/wspr-recorder"
SERVICE_USER="wsprrec"
SERVICE_GROUP="wsprrec"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
    fi
}

check_dependencies() {
    info "Checking dependencies..."
    
    # Check for Python 3.9+
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 9 ]]; then
        error "Python 3.9+ is required (found $PYTHON_VERSION)"
    fi
    
    info "Found Python $PYTHON_VERSION"
    
    # Check for venv module
    if ! python3 -c "import venv" &> /dev/null; then
        error "Python venv module is required. Install with: apt install python3-venv"
    fi
    
    # Check for pip
    if ! python3 -c "import pip" &> /dev/null; then
        error "Python pip is required. Install with: apt install python3-pip"
    fi
}

create_user() {
    info "Creating service user..."
    
    if id "$SERVICE_USER" &>/dev/null; then
        info "User $SERVICE_USER already exists"
    else
        useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
        info "Created user $SERVICE_USER"
    fi
    
    # Add user to audio group for potential future audio device access
    if getent group audio &>/dev/null; then
        usermod -a -G audio "$SERVICE_USER" 2>/dev/null || true
    fi
}

check_pattern_a() {
    # Contract v0.4 §12.5: the repo must be traversable by the service
    # user. The canonical location is /opt/git/wspr-recorder (group-readable);
    # repos under a mode-700 home directory are unreachable even if
    # individual files are world-readable.
    local repo_root
    repo_root="$(cd "$(dirname "$0")" && pwd)"
    local marker="$repo_root/wspr_recorder/__init__.py"
    if ! sudo -u "$SERVICE_USER" test -r "$marker"; then
        error "Service user $SERVICE_USER cannot read $marker — Pattern A violation.
    Fix: place the repo at /opt/git/wspr-recorder (not under a mode-700 home),
    or: chmod g+rx the path and add $SERVICE_USER to the owner's group."
    fi
    info "Pattern A check passed ($SERVICE_USER can traverse $repo_root)"
}

install_application() {
    info "Installing application to $INSTALL_DIR..."

    check_pattern_a

    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Check if this is an upgrade
    local IS_UPGRADE=false
    if [[ -d "$INSTALL_DIR/venv" ]]; then
        IS_UPGRADE=true
        info "Existing installation detected - performing upgrade"
    fi
    
    # Create or upgrade virtual environment
    if [[ "$IS_UPGRADE" == true ]]; then
        info "Upgrading virtual environment..."
        python3 -m venv --upgrade "$INSTALL_DIR/venv"
    else
        info "Creating virtual environment..."
        python3 -m venv "$INSTALL_DIR/venv"
    fi
    
    # Upgrade pip and wheel
    "$INSTALL_DIR/venv/bin/pip" install --upgrade pip wheel
    
    # Install/upgrade the package and all dependencies
    info "Installing/upgrading wspr-recorder and dependencies..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ "$IS_UPGRADE" == true ]]; then
        # Force upgrade of all dependencies including ka9q-python
        "$INSTALL_DIR/venv/bin/pip" install --upgrade --force-reinstall "$SCRIPT_DIR"
    else
        "$INSTALL_DIR/venv/bin/pip" install "$SCRIPT_DIR"
    fi
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
    
    if [[ "$IS_UPGRADE" == true ]]; then
        info "Application upgraded"
    else
        info "Application installed"
    fi
}

install_config() {
    info "Installing configuration..."
    
    mkdir -p "$CONFIG_DIR"
    
    # Install config file if it doesn't exist
    if [[ ! -f "$CONFIG_DIR/config.toml" ]]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        
        # Use example config as base, adjust paths
        if [[ -f "$SCRIPT_DIR/config.toml.example" ]]; then
            cp "$SCRIPT_DIR/config.toml.example" "$CONFIG_DIR/config.toml"
        else
            cp "$SCRIPT_DIR/config.toml" "$CONFIG_DIR/config.toml"
        fi
        
        # Update paths in config
        sed -i "s|output_dir = .*|output_dir = \"$OUTPUT_DIR\"|" "$CONFIG_DIR/config.toml"
        sed -i "s|ipc_socket = .*|ipc_socket = \"$RUN_DIR/control.sock\"|" "$CONFIG_DIR/config.toml"
        
        info "Installed default configuration to $CONFIG_DIR/config.toml"
        warn "Edit $CONFIG_DIR/config.toml to configure radiod address and frequencies"
    else
        info "Configuration already exists at $CONFIG_DIR/config.toml"
    fi
    
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$CONFIG_DIR"
    chmod 640 "$CONFIG_DIR/config.toml"
}

install_systemd() {
    info "Installing systemd service..."

    # Create tmpfiles.d config for runtime + log directories
    cat > /etc/tmpfiles.d/wspr-recorder.conf << EOF
# wspr-recorder runtime, output, and log directories
d $RUN_DIR 0755 $SERVICE_USER $SERVICE_GROUP -
d $OUTPUT_DIR 0755 $SERVICE_USER $SERVICE_GROUP -
d $LOG_DIR 0755 $SERVICE_USER $SERVICE_GROUP -
EOF

    systemd-tmpfiles --create /etc/tmpfiles.d/wspr-recorder.conf

    # Install the canonical templated unit from the repo (NOT an inline
    # unit). This keeps the deployed service in lockstep with
    # systemd/wspr-recorder@.service (Type=notify, WatchdogSec=180,
    # MemoryMax=1G, MALLOC_ARENA_MAX=2, EnvironmentFile coordination.env)
    # so sigmond-driven and standalone installs share one unit file.
    local repo_root unit_src unit_dst
    repo_root="$(cd "$(dirname "$0")" && pwd)"
    unit_src="$repo_root/systemd/wspr-recorder@.service"
    unit_dst="/etc/systemd/system/wspr-recorder@.service"

    if [[ ! -f "$unit_src" ]]; then
        error "Canonical unit not found at $unit_src"
    fi

    # Clean up the legacy non-templated unit from pre-contract installs.
    if [[ -f /etc/systemd/system/wspr-recorder.service ]]; then
        warn "Removing legacy non-templated /etc/systemd/system/wspr-recorder.service"
        systemctl disable --now wspr-recorder.service 2>/dev/null || true
        rm -f /etc/systemd/system/wspr-recorder.service
    fi

    ln -sfn "$unit_src" "$unit_dst"
    info "Installed $unit_dst -> $unit_src"

    systemctl daemon-reload

    info "Systemd service installed (templated). Enable with:"
    info "  sudo systemctl enable --now wspr-recorder@<instance>"
    info "where <instance> is derived from [radiod].status_address in config.toml"
    info "(e.g. bee3-status.local -> wspr-recorder@bee3)."
}

install_symlinks() {
    info "Creating command symlinks..."
    
    # Create symlinks in /usr/local/bin
    ln -sf "$INSTALL_DIR/venv/bin/wspr-recorder" /usr/local/bin/wspr-recorder
    ln -sf "$INSTALL_DIR/venv/bin/wspr-ctl" /usr/local/bin/wspr-ctl
    
    info "Commands available: wspr-recorder, wspr-ctl"
}

uninstall() {
    info "Uninstalling wspr-recorder..."
    
    # Stop and disable every templated instance, then the legacy unit.
    local instances
    instances=$(systemctl list-units --no-legend --all 'wspr-recorder@*.service' 2>/dev/null | awk '{print $1}')
    for inst in $instances; do
        systemctl disable --now "$inst" 2>/dev/null || true
    done
    systemctl disable --now wspr-recorder.service 2>/dev/null || true

    # Remove systemd files
    rm -f /etc/systemd/system/wspr-recorder.service
    rm -f /etc/systemd/system/wspr-recorder@.service
    rm -f /etc/tmpfiles.d/wspr-recorder.conf
    systemctl daemon-reload
    
    # Remove symlinks
    rm -f /usr/local/bin/wspr-recorder
    rm -f /usr/local/bin/wspr-ctl
    
    # Remove installation directory
    rm -rf "$INSTALL_DIR"
    
    # Remove runtime directories
    rm -rf "$RUN_DIR"
    rm -rf "$OUTPUT_DIR"
    
    # Optionally remove config (ask user)
    if [[ -d "$CONFIG_DIR" ]]; then
        read -p "Remove configuration in $CONFIG_DIR? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$CONFIG_DIR"
            info "Configuration removed"
        else
            info "Configuration preserved in $CONFIG_DIR"
        fi
    fi
    
    # Optionally remove user
    if id "$SERVICE_USER" &>/dev/null; then
        read -p "Remove service user $SERVICE_USER? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            userdel "$SERVICE_USER" 2>/dev/null || true
            info "User removed"
        fi
    fi
    
    info "Uninstallation complete"
}

show_status() {
    # "Upgrade" if any templated instance is already installed (enabled
    # or disabled but present). The unit is templated so we look for
    # enabled instances.
    local existing
    existing=$(systemctl list-unit-files --no-legend 'wspr-recorder@*.service' 2>/dev/null | awk '{print $1}' | head -1)
    local IS_UPGRADE=false
    if [[ -n "$existing" ]]; then
        IS_UPGRADE=true
    fi

    echo ""
    if [[ "$IS_UPGRADE" == true ]]; then
        echo "=============================================="
        echo "  wspr-recorder upgrade complete"
        echo "=============================================="
    else
        echo "=============================================="
        echo "  wspr-recorder installation complete"
        echo "=============================================="
    fi
    echo ""
    echo "Installation directory: $INSTALL_DIR"
    echo "Configuration file:     $CONFIG_DIR/config.toml"
    echo "IPC socket:             $RUN_DIR/control.sock"
    echo "WAV output:             $OUTPUT_DIR/<band>/"
    echo "Logs:                   $LOG_DIR/<instance>.log"
    echo ""
    echo "Instance name is derived from [radiod].status_address with"
    echo "-status.local / .local stripped. Example:"
    echo "  status_address = \"bee3-status.local\"  ->  wspr-recorder@bee3"
    echo ""
    if [[ "$IS_UPGRADE" == true ]]; then
        echo "Next steps:"
        echo "  1. Restart running instances to pick up the new version:"
        echo "     sudo systemctl restart 'wspr-recorder@*'"
        echo ""
        echo "  2. Check status:"
        echo "     sudo systemctl status wspr-recorder@<instance>"
        echo "     wspr-ctl health"
    else
        echo "Next steps:"
        echo "  1. Edit configuration:"
        echo "     sudoedit $CONFIG_DIR/config.toml"
        echo ""
        echo "  2. Validate it:"
        echo "     wspr-recorder validate --json"
        echo ""
        echo "  3. Enable and start your instance:"
        echo "     sudo systemctl enable --now wspr-recorder@<instance>"
        echo ""
        echo "  4. Check status:"
        echo "     sudo systemctl status wspr-recorder@<instance>"
        echo "     journalctl -fu wspr-recorder@<instance>"
        echo "     wspr-ctl health"
    fi
    echo ""
}

# Main
check_root

if [[ "$1" == "--uninstall" ]]; then
    uninstall
    exit 0
fi

check_dependencies
create_user
install_application
install_config
install_systemd
install_symlinks
show_status
