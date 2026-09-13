"""Quizbox IP Monitor — prints your IPv4 addresses every 60 seconds."""

import subprocess
import re
import time
import sys
from datetime import datetime


def get_ipv4_addresses():
    """Parse ipconfig output to extract IPv4 addresses for Ethernet and Wi-Fi."""
    try:
        output = subprocess.check_output("ipconfig", text=True, creationflags=0x08000000)
    except Exception:
        return []

    results = []
    current_adapter = None

    for line in output.splitlines():
        adapter_match = re.match(r"(?:Ethernet|Wi-Fi|Wireless).*adapter\s+(.+?):", line, re.I)
        if adapter_match:
            current_adapter = adapter_match.group(0).rstrip(":")
            continue

        ipv4_match = re.search(r"IPv4 Address[.\s]+:\s+([\d.]+)", line, re.I)
        if ipv4_match and current_adapter:
            addr = ipv4_match.group(1)
            if not addr.startswith("127."):
                results.append((current_adapter, addr))
            current_adapter = None

    return results


def main():
    print("\n  Quizbox IP Monitor — refreshing every 60s (Ctrl+C to stop)\n")
    try:
        while True:
            addrs = get_ipv4_addresses()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if addrs:
                header = f"  {'Adapter':<35} {'IPv4 Address'}"
                print(header)
                print(f"  {'─' * 35} {'─' * 17}")
                for name, ip in addrs:
                    print(f"  {name:<35} {ip}")
            else:
                print("  No network adapters found.")

            print(f"\n  Last updated: {now}\n")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n  Stopped.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
