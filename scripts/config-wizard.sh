#!/bin/bash
#
# wspr-recorder config wizard (whiptail).
#
# Called by `wspr-recorder config init` and `wspr-recorder config edit`
# when stdout is a TTY and whiptail is installed.  The single editable
# field in [radiod] is `status_address` (mDNS hostname of the radiod to
# join multicast from); the [[band]] array-of-tables is left to $EDITOR
# since whiptail can't naturally express it.
#
# UX inspiration: psk-recorder/scripts/config-wizard.sh.  This wizard is
# deliberately smaller — wspr-recorder's editable surface is essentially
# one field, so the menu is the wizard.
#
# Output protocol (consumed by configurator._exec_wizard):
#   STATUS_ADDRESS=<value>    on stdout → Python writes it via
#                              _replace_radiod_field() in the live TOML
#   (empty stdout)             → operator cancelled or used $EDITOR
#                              escape; Python reports "no changes"
#   non-zero exit              → real error; Python falls back to the
#                              legacy stdin path
#
# Environment:
#   WSPR_RECORDER_CONFIG   path to the live config.toml (default
#                          /etc/wspr-recorder/config.toml)
#   SIGMOND_ENV_CACHE      path to sigmond's environment cache
#                          (default /var/lib/sigmond/environment-cache.json)
#
set -euo pipefail

CONFIG_PATH="${WSPR_RECORDER_CONFIG:-/etc/wspr-recorder/config.toml}"
CACHE_PATH="${SIGMOND_ENV_CACHE:-/var/lib/sigmond/environment-cache.json}"

# --- shared shell helpers ---------------------------------------------------
#
# Source sigmond's Tier-1 wizard helpers (preflight_or_exit_2, _info /
# _warn / _err, recommended HEIGHT/WIDTH/LIST_HEIGHT/BACKTITLE
# defaults).  Python's _exec_wizard (wspr_recorder/configurator.py) sets
# SIGMOND_WIZARD_LIB_SH in the env via sigmond.wizard_dispatch when
# the lib is installed; the :- default below covers direct-invocation
# safety.
#
# Inline-fallback block keeps this script working when sigmond's
# library isn't on the host -- same behaviour as before the
# extraction.
SIGMOND_WIZARD_LIB_SH="${SIGMOND_WIZARD_LIB_SH:-/opt/git/sigmond/sigmond/lib/sigmond/wizard_dispatch/wizard_dispatch.sh}"
if [[ -r "$SIGMOND_WIZARD_LIB_SH" ]]; then
    # shellcheck disable=SC1090
    . "$SIGMOND_WIZARD_LIB_SH"
else
    # Local fallback (verbatim from pre-extraction shape so behaviour
    # is identical regardless of which path runs).
    HEIGHT=20; WIDTH=78; LIST_HEIGHT=10
    _info() { printf '  %s\n'                "$*" >&2; }
    _warn() { printf '  \033[33m⚠\033[0m %s\n' "$*" >&2; }
    _err()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; }
    preflight_or_exit_2() {
        command -v whiptail >/dev/null 2>&1 \
            || { _err "whiptail not on PATH"; exit 2; }
        [[ -t 1 ]] \
            || { _err "stdout is not a TTY"; exit 2; }
    }
fi

# Override BACKTITLE to client-specific (lib default is generic).
BACKTITLE="wspr-recorder configuration"

# --- whiptail sanity --------------------------------------------------------
# Belt-and-braces: Python's is_wizard_available() already gated for both
# conditions before exec'ing us, but operators sometimes invoke this
# script directly during dev.  Exit 2 distinguishes "shouldn't have
# been called" from operator-cancel (exit 0) and real error (exit 1).
preflight_or_exit_2

# --- current status_address extraction --------------------------------------
# tomllib via python is the most reliable parser; bash-side regex breaks on
# legitimate comments containing the key name.

current_status=""
if [[ -r "$CONFIG_PATH" ]]; then
    current_status=$(python3 - "$CONFIG_PATH" <<'PY' 2>/dev/null || true
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # noqa
with open(sys.argv[1], "rb") as f:
    data = tomllib.load(f)
print((data.get("radiod") or {}).get("status_address", ""))
PY
)
fi

# --- LAN radiod menu items from sigmond's environment cache -----------------
# Returns: blank-separated lines of "<tag>|<status_address>|<friendly_name>"
# where <tag> is short numeric or alphabetic key whiptail uses internally.

cache_lines=()
if [[ -r "$CACHE_PATH" ]]; then
    while IFS= read -r line; do
        [[ -n "$line" ]] && cache_lines+=("$line")
    done < <(python3 - "$CACHE_PATH" <<'PY' 2>/dev/null || true
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
seen = set()
for obs in data.get("observations") or []:
    if obs.get("source") != "mdns" or obs.get("kind") != "radiod" or not obs.get("ok", True):
        continue
    endpoint = obs.get("endpoint") or ""
    # endpoint format: "hostname.local:5006" → strip port for the address
    address = endpoint.rsplit(":", 1)[0] if ":" in endpoint else endpoint
    if not address or address in seen:
        continue
    seen.add(address)
    fields = obs.get("fields") or {}
    # mdns_name is the human-readable advertised name (e.g.
    # "AC0G @EM38ww B1 T3FD"); fall back to the bare hostname.
    label = (fields.get("mdns_name") or fields.get("name") or address).strip()
    print(f"{address}|{label}")
PY
)
fi

# --- build the whiptail menu ------------------------------------------------

menu_args=()
i=0
declare -A tag_to_address=()
for line in "${cache_lines[@]}"; do
    address="${line%%|*}"
    label="${line#*|}"
    tag="cached-$i"
    tag_to_address["$tag"]="$address"
    # Description column shows label + cached marker + (current) if it matches
    desc="$label"
    if [[ "$address" == "$current_status" ]]; then
        desc="$desc  (current)"
    else
        desc="$desc  (cached)"
    fi
    menu_args+=("$tag" "$desc")
    i=$((i + 1))
done

# If current_status isn't in the cache, surface it explicitly so the operator
# can re-pick it.
if [[ -n "$current_status" && -z "${tag_to_address[cached-current]:-}" ]]; then
    found_current=0
    for tag in "${!tag_to_address[@]}"; do
        if [[ "${tag_to_address[$tag]}" == "$current_status" ]]; then
            found_current=1; break
        fi
    done
    if [[ $found_current -eq 0 ]]; then
        tag_to_address["cached-current"]="$current_status"
        menu_args+=("cached-current" "$current_status  (current; not in cache)")
    fi
fi

menu_args+=("manual"    "Enter another status address by hand…")
menu_args+=("edit-toml" "Open $CONFIG_PATH in \$EDITOR…")
menu_args+=("cancel"    "Cancel (no changes)")

# --- prompt -----------------------------------------------------------------

if [[ ${#cache_lines[@]} -eq 0 ]]; then
    _info "No radiod instances in the sigmond environment cache."
    _info "Run \`smd environment probe\` to populate it, or enter manually."
fi

selection=$(whiptail \
    --title "wspr-recorder: choose radiod status_address" \
    --backtitle "$BACKTITLE" \
    --menu "Pick the radiod multicast control plane this recorder should join.\nLAN entries come from sigmond's environment cache." \
    "$HEIGHT" "$WIDTH" "$LIST_HEIGHT" \
    "${menu_args[@]}" \
    3>&1 1>&2 2>&3) || {
    # User hit ESC or Cancel — treat as cancel.
    exit 0
}

case "$selection" in
    cancel)
        exit 0
        ;;
    edit-toml)
        editor="${VISUAL:-${EDITOR:-vi}}"
        "$editor" "$CONFIG_PATH" >&2
        exit 0
        ;;
    manual)
        new_address=$(whiptail \
            --title "wspr-recorder: status address" \
            --backtitle "$BACKTITLE" \
            --inputbox "Enter the radiod status address (mDNS hostname, e.g. bee1-status.local):" \
            12 "$WIDTH" "$current_status" \
            3>&1 1>&2 2>&3) || exit 0
        new_address="${new_address//[[:space:]]/}"
        if [[ -z "$new_address" ]]; then
            _warn "empty address; no changes"
            exit 0
        fi
        printf 'STATUS_ADDRESS=%s\n' "$new_address"
        exit 0
        ;;
    cached-*)
        chosen="${tag_to_address[$selection]:-}"
        if [[ -z "$chosen" ]]; then
            _err "internal: tag $selection has no mapped address"
            exit 1
        fi
        printf 'STATUS_ADDRESS=%s\n' "$chosen"
        exit 0
        ;;
    *)
        _err "unrecognized selection: $selection"
        exit 1
        ;;
esac
