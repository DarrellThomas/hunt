#!/usr/bin/env python3
"""
Detect monitor resolution for HUNT simulation.

Usage:
    python detect_monitor.py

Output:
    Prints the native monitor resolution and recommended settings.
"""

import sys

try:
    import pygame
    pygame.init()

    # Get display info
    display_info = pygame.display.Info()
    width = display_info.current_w
    height = display_info.current_h

    print("="*60)
    print("HUNT Monitor Detection")
    print("="*60)
    print(f"\nNative Resolution: {width} x {height}")

    # Calculate aspect ratio
    from math import gcd
    divisor = gcd(width, height)
    aspect_w = width // divisor
    aspect_h = height // divisor
    print(f"Aspect Ratio: {aspect_w}:{aspect_h}")

    # Recommend settings
    print("\n" + "="*60)
    print("Recommended Settings")
    print("="*60)

    print("\nFor GPU (fullscreen):")
    print(f"  python src/main_gpu.py --width {width} --height {height}")
    print(f"  or modify main_gpu.py:")
    print(f"    width={width}, height={height}")

    print("\nFor CPU (windowed):")
    # Recommend half resolution for CPU
    cpu_width = width // 2
    cpu_height = height // 2
    print(f"  python src/main.py --width {cpu_width} --height {cpu_height}")
    print(f"  or modify main.py:")
    print(f"    width={cpu_width}, height={cpu_height}")

    # Common resolutions
    print("\n" + "="*60)
    print("Common Resolutions")
    print("="*60)
    resolutions = [
        ("1920x1080", "Full HD (16:9)"),
        ("2560x1440", "QHD (16:9)"),
        ("3840x2160", "4K UHD (16:9)"),
        ("1600x1200", "UXGA (4:3)"),
        ("2560x1600", "WQXGA (16:10)"),
    ]

    for res, desc in resolutions:
        marker = " ← YOUR MONITOR" if res == f"{width}x{height}" else ""
        print(f"  {res:<12} {desc}{marker}")

    pygame.quit()

except ImportError:
    print("Error: pygame is not installed")
    print("Install with: pip install pygame")
    sys.exit(1)
except Exception as e:
    print(f"Error detecting monitor: {e}")
    sys.exit(1)
