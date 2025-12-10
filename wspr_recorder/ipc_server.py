"""
IPC Server for wspr-recorder.

Provides a Unix domain socket interface for external tools (wsprdaemon)
to query status and control the recorder.

Protocol: JSON-RPC 2.0 style over Unix socket
- Request: {"method": "...", "params": {...}, "id": 1}
- Response: {"result": {...}, "id": 1} or {"error": {...}, "id": 1}

Simple queries can omit "params" and "id".
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class IPCError(Exception):
    """IPC error with code and message."""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class IPCServer:
    """
    Unix domain socket IPC server.
    
    Provides a simple RPC interface for wsprdaemon and other tools
    to query and control wspr-recorder.
    
    Usage:
        server = IPCServer("/run/wspr-recorder/control.sock")
        server.register("status", lambda params: {"running": True})
        await server.start()
        ...
        await server.stop()
    """
    
    def __init__(
        self,
        socket_path: str,
        permissions: int = 0o660,
    ):
        """
        Initialize IPC server.
        
        Args:
            socket_path: Path to Unix domain socket
            permissions: Socket file permissions (default: owner+group rw)
        """
        self.socket_path = Path(socket_path)
        self.permissions = permissions
        
        self._server: Optional[asyncio.AbstractServer] = None
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        
        # Register built-in methods
        self.register("ping", self._handle_ping)
        self.register("list_methods", self._handle_list_methods)
    
    def register(self, method: str, handler: Callable[[Optional[Dict]], Any]) -> None:
        """
        Register a method handler.
        
        Args:
            method: Method name
            handler: Callable that takes optional params dict and returns result
        """
        self._handlers[method] = handler
        logger.debug(f"IPC: Registered method '{method}'")
    
    def unregister(self, method: str) -> None:
        """Unregister a method handler."""
        self._handlers.pop(method, None)
    
    async def start(self) -> None:
        """Start the IPC server."""
        if self._running:
            return
        
        # Ensure parent directory exists
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove stale socket file
        if self.socket_path.exists():
            self.socket_path.unlink()
        
        # Create server
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
        )
        
        # Set permissions
        os.chmod(self.socket_path, self.permissions)
        
        self._running = True
        logger.info(f"IPC server listening on {self.socket_path}")
    
    async def stop(self) -> None:
        """Stop the IPC server."""
        if not self._running:
            return
        
        self._running = False
        
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        
        # Clean up socket file
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except Exception:
                pass
        
        logger.info("IPC server stopped")
    
    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection."""
        peer = "unknown"
        try:
            # Read request (single line JSON)
            data = await asyncio.wait_for(
                reader.readline(),
                timeout=5.0,
            )
            
            if not data:
                return
            
            # Parse and handle request
            response = await self._process_request(data.decode('utf-8').strip())
            
            # Send response
            writer.write((json.dumps(response) + '\n').encode('utf-8'))
            await writer.drain()
            
        except asyncio.TimeoutError:
            logger.debug(f"IPC: Client timeout")
        except Exception as e:
            logger.warning(f"IPC: Client error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
    
    async def _process_request(self, data: str) -> Dict[str, Any]:
        """Process a JSON-RPC request."""
        request_id = None
        
        try:
            # Parse JSON
            try:
                request = json.loads(data)
            except json.JSONDecodeError as e:
                raise IPCError(PARSE_ERROR, f"Parse error: {e}")
            
            # Validate request
            if not isinstance(request, dict):
                raise IPCError(INVALID_REQUEST, "Request must be an object")
            
            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})
            
            if not method:
                raise IPCError(INVALID_REQUEST, "Missing 'method'")
            
            if not isinstance(method, str):
                raise IPCError(INVALID_REQUEST, "'method' must be a string")
            
            if params is not None and not isinstance(params, dict):
                raise IPCError(INVALID_PARAMS, "'params' must be an object")
            
            # Find handler
            handler = self._handlers.get(method)
            if not handler:
                raise IPCError(METHOD_NOT_FOUND, f"Method not found: {method}")
            
            # Call handler
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(params)
                else:
                    result = handler(params)
            except IPCError:
                raise
            except Exception as e:
                logger.exception(f"IPC: Handler error for '{method}'")
                raise IPCError(INTERNAL_ERROR, str(e))
            
            # Build response
            response = {"result": result}
            if request_id is not None:
                response["id"] = request_id
            
            return response
            
        except IPCError as e:
            response = {
                "error": {
                    "code": e.code,
                    "message": e.message,
                }
            }
            if request_id is not None:
                response["id"] = request_id
            return response
    
    # Built-in handlers
    
    def _handle_ping(self, params: Optional[Dict]) -> Dict[str, Any]:
        """Handle ping request."""
        return {"pong": True}
    
    def _handle_list_methods(self, params: Optional[Dict]) -> Dict[str, Any]:
        """List available methods."""
        return {"methods": sorted(self._handlers.keys())}


class IPCClient:
    """
    Simple IPC client for testing and scripting.
    
    Usage:
        client = IPCClient("/run/wspr-recorder/control.sock")
        result = await client.call("status")
    """
    
    def __init__(self, socket_path: str, timeout: float = 5.0):
        """
        Initialize IPC client.
        
        Args:
            socket_path: Path to Unix domain socket
            timeout: Request timeout in seconds
        """
        self.socket_path = socket_path
        self.timeout = timeout
    
    async def call(
        self,
        method: str,
        params: Optional[Dict] = None,
        request_id: Optional[int] = None,
    ) -> Any:
        """
        Call an IPC method.
        
        Args:
            method: Method name
            params: Optional parameters
            request_id: Optional request ID
            
        Returns:
            Result from the method
            
        Raises:
            IPCError: On RPC error
            Exception: On connection/timeout error
        """
        # Build request
        request: Dict[str, Any] = {"method": method}
        if params:
            request["params"] = params
        if request_id is not None:
            request["id"] = request_id
        
        # Connect and send
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(self.socket_path),
            timeout=self.timeout,
        )
        
        try:
            writer.write((json.dumps(request) + '\n').encode('utf-8'))
            await writer.drain()
            
            # Read response
            data = await asyncio.wait_for(
                reader.readline(),
                timeout=self.timeout,
            )
            
            response = json.loads(data.decode('utf-8'))
            
            if "error" in response:
                error = response["error"]
                raise IPCError(error.get("code", -1), error.get("message", "Unknown error"))
            
            return response.get("result")
            
        finally:
            writer.close()
            await writer.wait_closed()


def ipc_query(socket_path: str, method: str, params: Optional[Dict] = None) -> Any:
    """
    Synchronous IPC query helper.
    
    Convenience function for simple queries from scripts.
    
    Args:
        socket_path: Path to Unix domain socket
        method: Method name
        params: Optional parameters
        
    Returns:
        Result from the method
    """
    async def _query():
        client = IPCClient(socket_path)
        return await client.call(method, params)
    
    return asyncio.run(_query())
