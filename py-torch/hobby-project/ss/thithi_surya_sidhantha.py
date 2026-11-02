"""
Live Tithi display using Surya Siddhanta.
Updates every second. Uses your location for local time display.

Usage:
    python3 tithi_live.py                        # defaults to Dublin
    python3 tithi_live.py --lat 53.35 --lon -6.26  # custom location
"""

import math
import time
import sys
import os
import argparse
from datetime import datetime, timezone, timedelta

# ── Surya Siddhanta constants ──────────────────────────────────────
CIVIL_DAYS     = 1_577_917_828
REV_SUN        = 4_320_000
REV_MOON       = 57_753_336
REV_APOGEE     = 488_219
REV_NODE       = 232_226

SUN_MANDA_MAX  = 2 + 10/60 + 32/3600
MOON_MANDA_MAX = 4 + 56/60
SUN_APOGEE     = 77.5

TITHI_NAMES = [
    "Pratipada","Dwitiya","Tritiya","Chaturthi","Panchami",
    "Shashthi","Saptami","Ashtami","Navami","Dashami",
    "Ekadashi","Dwadashi","Trayodashi","Chaturdashi","Purnima",
    "Pratipada","Dwitiya","Tritiya","Chaturthi","Panchami",
    "Shashthi","Saptami","Ashtami","Navami","Dashami",
    "Ekadashi","Dwadashi","Trayodashi","Chaturdashi","Amavasya"
]

# ── Core calculations ──────────────────────────────────────────────
def julian_day(y, m, d, h=0.5):
    if m <= 2:
        y -= 1; m += 12
    A = y // 100
    B = 2 - A + A // 4
    return int(365.25*(y+4716)) + int(30.6001*(m+1)) + d + h/24 + B - 1524.5

def moon_sun_diff(days_since_kali):
    dm_sun    = REV_SUN    * 360.0 / CIVIL_DAYS
    dm_moon   = REV_MOON   * 360.0 / CIVIL_DAYS
    dm_apogee = REV_APOGEE * 360.0 / CIVIL_DAYS
    dm_node   = REV_NODE   * 360.0 / CIVIL_DAYS

    sun_mean   = (dm_sun    * days_since_kali) % 360
    moon_mean  = (dm_moon   * days_since_kali) % 360
    apogee     = (dm_apogee * days_since_kali) % 360

    sun_eq  = SUN_MANDA_MAX  * math.sin(math.radians((sun_mean  - SUN_APOGEE) % 360))
    moon_eq = MOON_MANDA_MAX * math.sin(math.radians((moon_mean - apogee) % 360))

    sun_true  = (sun_mean  + sun_eq)  % 360
    moon_true = (moon_mean + moon_eq) % 360

    return (moon_true - sun_true) % 360

def calc(ut_hours, y, m, d, kali_jd):
    jd = julian_day(y, m, d, ut_hours)
    days = jd - kali_jd
    diff = moon_sun_diff(days)
    t_num = diff / 12
    t_idx = int(t_num)
    frac  = t_num - t_idx
    return t_idx, frac, diff

def find_tithi_end(t_idx, y, m, d, ut_hours, kali_jd):
    """Binary search for when current tithi ends (next 12° boundary)."""
    target_diff = (t_idx + 1) * 12.0

    # Search up to 2 days ahead in hours
    lo, hi = ut_hours, ut_hours + 48.0
    for _ in range(50):
        mid = (lo + hi) / 2
        # Handle day overflow
        total_hours = mid
        dd, hh = divmod(total_hours - ut_hours, 24)
        import math as _m
        from datetime import date
        future = datetime(y, m, d, tzinfo=timezone.utc) + timedelta(hours=mid - ut_hours + ut_hours)
        future = datetime(y, m, d, tzinfo=timezone.utc) + timedelta(hours=(mid - ut_hours))
        fy, fm, fd = future.year, future.month, future.day
        fh = future.hour + future.minute/60 + future.second/3600

        jd = julian_day(fy, fm, fd, fh + (mid - ut_hours - (future - datetime(y,m,d,tzinfo=timezone.utc)).total_seconds()/3600))
        # simpler: just pass absolute hours offset
        jd = julian_day(y, m, d, mid)
        days = jd - kali_jd
        diff = moon_sun_diff(days)
        # Unwrap near boundary
        t_num = diff / 12
        if t_num < t_idx:          # wrapped around 360
            t_num += 30
        if t_num >= t_idx + 1:
            hi = mid
        else:
            lo = mid
    end_ut = (lo + hi) / 2
    # Convert excess hours to actual datetime
    end_dt = datetime(y, m, d, tzinfo=timezone.utc) + timedelta(hours=(end_ut - ut_hours))
    return end_dt

# ── Sunrise (simple) ──────────────────────────────────────────────
def sunrise_utc(y, m, d, lat, lon):
    """Approximate sunrise in UTC hours for given date and location."""
    jd = julian_day(y, m, d, 12)
    n  = jd - 2451545.0
    L  = (280.46 + 0.9856474 * n) % 360
    g  = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2*g))
    dec = math.asin(math.sin(math.radians(23.439)) * math.sin(lam))
    lat_r = math.radians(lat)
    cos_ha = -math.tan(lat_r) * math.tan(dec)
    cos_ha = max(-1, min(1, cos_ha))
    ha = math.degrees(math.acos(cos_ha))
    noon_utc = 12 - lon/15
    sunrise  = noon_utc - ha/15
    return sunrise

# ── Display ───────────────────────────────────────────────────────
def display(t_idx, frac, diff, now_local, end_dt, local_tz_offset, location_name):
    paksha = "Shukla ☽" if t_idx < 15 else "Krishna 🌑"
    name   = TITHI_NAMES[t_idx]

    # Time remaining
    remaining = end_dt - datetime.now(timezone.utc)
    total_sec = remaining.total_seconds()
    if total_sec < 0:
        time_left = "ended"
    else:
        h = int(total_sec // 3600)
        m = int((total_sec % 3600) // 60)
        s = int(total_sec % 60)
        time_left = f"{h:02d}h {m:02d}m {s:02d}s"

    # End time in local
    end_local = end_dt + timedelta(hours=local_tz_offset)

    bar_len = 30
    filled  = int(frac * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)

    os.system("clear")
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │      SURYA SIDDHANTA TITHI (LIVE)       │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Location : {location_name:<29}│")
    print(f"  │  Time     : {now_local.strftime('%Y-%m-%d %H:%M:%S'):<29}│")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Paksha   : {paksha:<29}│")
    print(f"  │  Tithi    : {name:<29}│")
    print(f"  │  Number   : {t_idx+1}/30{'':<25}│")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Progress : [{bar}] {frac*100:4.1f}%  │")
    print(f"  │  Ends at  : {end_local.strftime('%H:%M:%S')} local{'':<20}│")
    print(f"  │  Remains  : {time_left:<29}│")
    print(f"  │  Moon-Sun : {diff:>8.4f}°{'':<20}│")
    print(f"  └─────────────────────────────────────────┘")
    print(f"  Press Ctrl+C to exit\n")

# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat",  type=float, default=53.3498, help="Latitude")
    parser.add_argument("--lon",  type=float, default=-6.2603, help="Longitude")
    parser.add_argument("--name", type=str,   default="Dublin, Ireland")
    args = parser.parse_args()

    # UTC offset from longitude (approximate; 15° per hour)
    tz_offset = args.lon / 15.0

    kali_jd = julian_day(-3101, 2, 18)

    print("  Starting live Tithi display... (Ctrl+C to stop)")
    time.sleep(1)

    while True:
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc + timedelta(hours=tz_offset)
        y, m, d = now_utc.year, now_utc.month, now_utc.day
        ut_h = now_utc.hour + now_utc.minute/60 + now_utc.second/3600

        t_idx, frac, diff = calc(ut_h, y, m, d, kali_jd)
        end_dt = find_tithi_end(t_idx, y, m, d, ut_h, kali_jd)

        display(t_idx, frac, diff, now_local, end_dt, tz_offset, args.name)
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Stopped.\n")