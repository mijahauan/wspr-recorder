#!/usr/bin/env python3
"""
wspr-ctl: Command-line interface for wspr-recorder IPC.

Query and control wspr-recorder from the command line or scripts.

Usage:
    wspr-ctl status          # Full status
    wspr-ctl health          # Quick health check
    wspr-ctl timing          # Timing information
    wspr-ctl bands           # List bands
    wspr-ctl band 20         # Status for specific band
    wspr-ctl config          # Configuration
    wspr-ctl ping            # Check if running
    wspr-ctl methods         # List available methods
    wspr-ctl call <method> [json_params]  # Raw method call
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .ipc_server import IPCClient, IPCError

DEFAULT_SOCKET = "/run/wspr-recorder/control.sock"


async def call_method(socket_path: str, method: str, params: dict = None) -> dict:
    """Call an IPC method and return the result."""
    client = IPCClient(socket_path)
    return await client.call(method, params)


def format_output(data, compact: bool = False) -> str:
    """Format output data as JSON."""
    if compact:
        return json.dumps(data)
    return json.dumps(data, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="wspr-recorder control interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "-s", "--socket",
        default=DEFAULT_SOCKET,
        help=f"IPC socket path (default: {DEFAULT_SOCKET})",
    )
    parser.add_argument(
        "-c", "--compact",
        action="store_true",
        help="Compact JSON output (single line)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - only output result, no errors",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Simple commands (no arguments)
    subparsers.add_parser("status", help="Get full status")
    subparsers.add_parser("health", help="Quick health check")
    subparsers.add_parser("timing", help="Get timing information")
    subparsers.add_parser("bands", help="List configured bands")
    subparsers.add_parser("config", help="Get configuration")
    subparsers.add_parser("ping", help="Check if recorder is running")
    subparsers.add_parser("methods", help="List available IPC methods")
    
    # Band status (requires band name)
    band_parser = subparsers.add_parser("band", help="Get status for specific band")
    band_parser.add_argument("band_name", help="Band name (e.g., 20, 40, 80eu)")
    
    # Raw method call
    call_parser = subparsers.add_parser("call", help="Call arbitrary IPC method")
    call_parser.add_argument("method", help="Method name")
    call_parser.add_argument("params", nargs="?", default="{}", help="JSON parameters")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Map commands to methods
    method_map = {
        "status": ("status", None),
        "health": ("health", None),
        "timing": ("timing", None),
        "bands": ("bands", None),
        "config": ("config", None),
        "ping": ("ping", None),
        "methods": ("list_methods", None),
    }
    
    try:
        if args.command in method_map:
            method, params = method_map[args.command]
        elif args.command == "band":
            method = "band_status"
            params = {"band": args.band_name}
        elif args.command == "call":
            method = args.method
            try:
                params = json.loads(args.params) if args.params != "{}" else None
            except json.JSONDecodeError as e:
                if not args.quiet:
                    print(f"Invalid JSON params: {e}", file=sys.stderr)
                return 1
        else:
            parser.print_help()
            return 1
        
        # Make the call
        result = asyncio.run(call_method(args.socket, method, params))
        print(format_output(result, args.compact))
        
        # For health command, return non-zero if unhealthy
        if args.command == "health" and isinstance(result, dict):
            if not result.get("healthy", True):
                return 2
        
        return 0
        
    except IPCError as e:
        if not args.quiet:
            print(json.dumps({"error": {"code": e.code, "message": e.message}}))
        return 1
    except FileNotFoundError:
        if not args.quiet:
            print(json.dumps({"error": "Socket not found - is wspr-recorder running?"}))
        return 1
    except ConnectionRefusedError:
        if not args.quiet:
            print(json.dumps({"error": "Connection refused - is wspr-recorder running?"}))
        return 1
    except asyncio.TimeoutError:
        if not args.quiet:
            print(json.dumps({"error": "Timeout connecting to wspr-recorder"}))
        return 1
    except Exception as e:
        if not args.quiet:
            print(json.dumps({"error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
