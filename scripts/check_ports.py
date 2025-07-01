#!/usr/bin/env python3
"""
Port Checker for PyNucleus Dashboard

Simple script to check which ports are available.
"""

import socket
import sys

def check_port(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def main():
    """Check common dashboard ports"""
    ports_to_check = [5000, 5001, 8080, 8000, 3000]
    
    print("🔍 Checking available ports for PyNucleus Dashboard...")
    print("-" * 50)
    
    available_ports = []
    
    for port in ports_to_check:
        if check_port(port):
            status = "✅ Available"
            available_ports.append(port)
        else:
            status = "❌ In Use"
        
        print(f"Port {port:4d}: {status}")
    
    print("-" * 50)
    
    if available_ports:
        print(f"🎉 Found {len(available_ports)} available port(s): {', '.join(map(str, available_ports))}")
        print(f"🚀 Dashboard will use port: {available_ports[0]}")
    else:
        print("❌ No ports available! Please free up a port or add more options.")
        sys.exit(1)

if __name__ == "__main__":
    main() 