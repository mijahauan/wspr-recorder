"""
RTP Ingestion for wspr-recorder.

Asyncio-based UDP receiver that:
- Receives RTP multicast packets
- Demultiplexes by SSRC
- Routes packets to appropriate BandRecorder instances
"""

import asyncio
import socket
import struct
import logging
import threading
from typing import Dict, Optional, Callable, NamedTuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class RTPHeader(NamedTuple):
    """Parsed RTP packet header."""
    version: int
    padding: bool
    extension: bool
    csrc_count: int
    marker: bool
    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int


@dataclass
class IngestStats:
    """Statistics for RTP ingestion."""
    packets_received: int = 0
    packets_routed: int = 0
    packets_unknown_ssrc: int = 0
    packets_invalid: int = 0
    bytes_received: int = 0
    
    def to_dict(self) -> dict:
        return {
            "packets_received": self.packets_received,
            "packets_routed": self.packets_routed,
            "packets_unknown_ssrc": self.packets_unknown_ssrc,
            "packets_invalid": self.packets_invalid,
            "bytes_received": self.bytes_received,
        }


def parse_rtp_header(data: bytes) -> Optional[RTPHeader]:
    """
    Parse RTP packet header.
    
    Args:
        data: Raw packet bytes (minimum 12 bytes)
    
    Returns:
        RTPHeader if valid, None if invalid
    """
    if len(data) < 12:
        return None
    
    # Parse RTP header (RFC 3550)
    byte0, byte1 = struct.unpack('!BB', data[0:2])
    
    version = (byte0 >> 6) & 0x03
    if version != 2:
        return None  # Only RTP version 2 supported
    
    padding = bool(byte0 & 0x20)
    extension = bool(byte0 & 0x10)
    csrc_count = byte0 & 0x0F
    
    marker = bool(byte1 & 0x80)
    payload_type = byte1 & 0x7F
    
    sequence, timestamp, ssrc = struct.unpack('!HIL', data[2:12])
    
    return RTPHeader(
        version=version,
        padding=padding,
        extension=extension,
        csrc_count=csrc_count,
        marker=marker,
        payload_type=payload_type,
        sequence=sequence,
        timestamp=timestamp,
        ssrc=ssrc
    )


def extract_payload(data: bytes, header: RTPHeader) -> bytes:
    """
    Extract payload from RTP packet.
    
    Args:
        data: Raw packet bytes
        header: Parsed RTP header
        
    Returns:
        Payload bytes
    """
    # Skip RTP header (12 bytes) + CSRC list (4 bytes each)
    offset = 12 + (4 * header.csrc_count)
    
    # Handle extension header if present
    if header.extension and len(data) > offset + 4:
        # Extension header: 2 bytes profile, 2 bytes length (in 32-bit words)
        ext_length = struct.unpack('!H', data[offset+2:offset+4])[0]
        offset += 4 + (ext_length * 4)
    
    # Handle padding if present
    payload_end = len(data)
    if header.padding and len(data) > 0:
        padding_length = data[-1]
        payload_end -= padding_length
    
    return data[offset:payload_end]


# Type for packet callback: (ssrc, header, payload) -> None
PacketCallback = Callable[[int, RTPHeader, bytes], None]


class RTPProtocol(asyncio.DatagramProtocol):
    """
    Asyncio protocol for receiving RTP packets.
    
    Parses headers and routes packets by SSRC to registered handlers.
    """
    
    def __init__(
        self,
        ssrc_handlers: Dict[int, PacketCallback],
        on_unknown_ssrc: Optional[Callable[[int], None]] = None,
        stats: Optional[IngestStats] = None,
    ):
        """
        Initialize RTP protocol.
        
        Args:
            ssrc_handlers: Dict mapping SSRC -> callback function
            on_unknown_ssrc: Optional callback for unknown SSRCs
            stats: Statistics object to update
        """
        self.ssrc_handlers = ssrc_handlers
        self.on_unknown_ssrc = on_unknown_ssrc
        self.stats = stats or IngestStats()
        self.transport: Optional[asyncio.DatagramTransport] = None
    
    def connection_made(self, transport: asyncio.DatagramTransport):
        """Called when connection is established."""
        self.transport = transport
        logger.info("RTP receiver connected")
    
    def datagram_received(self, data: bytes, addr):
        """
        Called when a datagram is received.
        
        This is the hot path - keep it fast!
        """
        self.stats.packets_received += 1
        self.stats.bytes_received += len(data)
        
        # Parse header
        header = parse_rtp_header(data)
        if header is None:
            self.stats.packets_invalid += 1
            return
        
        # Route by SSRC
        handler = self.ssrc_handlers.get(header.ssrc)
        if handler is not None:
            payload = extract_payload(data, header)
            try:
                handler(header.ssrc, header, payload)
                self.stats.packets_routed += 1
            except Exception as e:
                logger.error(f"Error in packet handler for SSRC {header.ssrc}: {e}")
        else:
            self.stats.packets_unknown_ssrc += 1
            if self.on_unknown_ssrc:
                self.on_unknown_ssrc(header.ssrc)
    
    def error_received(self, exc):
        """Called when a send/receive operation fails."""
        logger.error(f"RTP protocol error: {exc}")
    
    def connection_lost(self, exc):
        """Called when connection is lost."""
        if exc:
            logger.error(f"RTP connection lost: {exc}")
        else:
            logger.info("RTP connection closed")


class RTPIngest:
    """
    High-level RTP ingestion manager using threading.
    
    Uses a dedicated thread for packet reception (like grape-recorder)
    for reliable multicast handling.
    """
    
    def __init__(
        self,
        multicast_address: str,
        port: int,
    ):
        """
        Initialize RTP ingest.
        
        Args:
            multicast_address: Multicast group address (e.g., "239.1.2.3")
            port: UDP port number
        """
        self.multicast_address = multicast_address
        self.port = port
        
        self.ssrc_handlers: Dict[int, PacketCallback] = {}
        self.stats = IngestStats()
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._unknown_ssrcs: set = set()
    
    def register_handler(self, ssrc: int, handler: PacketCallback) -> None:
        """
        Register a handler for an SSRC.
        
        Args:
            ssrc: SSRC to handle
            handler: Callback function (ssrc, header, payload) -> None
        """
        self.ssrc_handlers[ssrc] = handler
        logger.debug(f"Registered handler for SSRC {ssrc}")
    
    def unregister_handler(self, ssrc: int) -> None:
        """
        Unregister a handler for an SSRC.
        
        Args:
            ssrc: SSRC to unregister
        """
        self.ssrc_handlers.pop(ssrc, None)
        logger.debug(f"Unregistered handler for SSRC {ssrc}")
    
    def _on_unknown_ssrc(self, ssrc: int) -> None:
        """Handle unknown SSRC (log once per SSRC)."""
        if ssrc not in self._unknown_ssrcs:
            self._unknown_ssrcs.add(ssrc)
            logger.debug(f"Unknown SSRC: {ssrc}")
    
    async def start(self) -> None:
        """
        Start receiving RTP packets.
        
        Starts a background thread for packet reception.
        """
        if self._running:
            logger.warning("RTP ingest already running")
            return
            
        logger.info(f"Starting RTP ingest on {self.multicast_address}:{self.port}")
        
        # Create UDP socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Request large receive buffer to prevent packet loss
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 26214400)
            actual_size = self._socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            logger.info(f"UDP receive buffer: requested 25MB, got {actual_size // 1024 // 1024}MB")
        except Exception as e:
            logger.warning(f"Could not set UDP buffer size: {e}")
        
        # Bind to port
        self._socket.bind(('', self.port))
        
        # Join multicast group - try loopback first (for local radiod)
        try:
            mreq = struct.pack(
                "4s4s",
                socket.inet_aton(self.multicast_address),
                socket.inet_aton('127.0.0.1')
            )
            self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            logger.info(f"Joined multicast {self.multicast_address} on loopback")
        except OSError:
            # Fallback to any interface
            mreq = struct.pack(
                "4sl",
                socket.inet_aton(self.multicast_address),
                socket.INADDR_ANY
            )
            self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            logger.info(f"Joined multicast {self.multicast_address} on all interfaces")
        
        # Start receiver thread
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        
        logger.info("RTP ingest started")
    
    def _receive_loop(self) -> None:
        """Main packet reception loop (runs in thread)."""
        packet_count = 0
        ssrc_seen = set()
        
        while self._running:
            try:
                data, addr = self._socket.recvfrom(8192)
                self.stats.packets_received += 1
                self.stats.bytes_received += len(data)
                
                # Parse RTP header
                header = parse_rtp_header(data)
                if header is None:
                    self.stats.packets_invalid += 1
                    continue
                
                # Log first packet from each SSRC
                if header.ssrc not in ssrc_seen:
                    ssrc_seen.add(header.ssrc)
                    has_handler = header.ssrc in self.ssrc_handlers
                    logger.debug(f"First packet from SSRC {header.ssrc}: "
                                f"handler={'YES' if has_handler else 'NO'}")
                
                # Extract payload
                payload = extract_payload(data, header)
                
                # Route by SSRC
                handler = self.ssrc_handlers.get(header.ssrc)
                if handler is not None:
                    try:
                        handler(header.ssrc, header, payload)
                        self.stats.packets_routed += 1
                    except Exception as e:
                        logger.error(f"Error in packet handler for SSRC {header.ssrc}: {e}")
                else:
                    self.stats.packets_unknown_ssrc += 1
                    self._on_unknown_ssrc(header.ssrc)
                
                # Log periodic stats
                packet_count += 1
                if packet_count % 10000 == 0:
                    logger.info(f"RTP: {packet_count} packets, "
                               f"{len(ssrc_seen)} SSRCs ({len(self.ssrc_handlers)} registered)")
                    
            except Exception as e:
                if self._running:
                    logger.error(f"Error receiving RTP packet: {e}")
    
    async def stop(self) -> None:
        """Stop receiving RTP packets."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._socket:
            self._socket.close()
        
        logger.info("RTP ingest stopped")
    
    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        return self.stats.to_dict()
    
    def clear_unknown_ssrcs(self) -> None:
        """Clear the set of unknown SSRCs (for re-logging)."""
        self._unknown_ssrcs.clear()
