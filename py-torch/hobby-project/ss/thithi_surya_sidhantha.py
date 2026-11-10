"""
Tithi calculator — compares Surya Siddhanta vs Modern (Drik) ephemeris.
Usage: python3 tithi_surya_siddhanta.py [YYYY-MM-DD]
       (defaults to today if no date given)
"""

import math
import sys
import ephem
from datetime import date, datetime, timezone

# ── Surya Siddhanta constants ──────────────────────────────────────
CIVIL_DAYS    = 1_577_917_828   # civil days per Mahayuga

REV_SUN       = 4_320_000
REV_MOON      = 57_753_336
REV_APOGEE    = 488_219         # Moon's mandocca (apogee)
REV_NODE      = 232_226         # Moon's ascending node (retrograde)

SUN_MANDA_MAX  = 2 + 10/60 + 32/3600   # 2°10'32"
MOON_MANDA_MAX = 4 + 56/60              # 4°56'
SUN_APOGEE     = 77.5                   # degrees (nearly fixed)

TITHI_NAMES = [
    "Pratipada","Dwitiya","Tritiya","Chaturthi","Panchami",
    "Shashthi","Saptami","Ashtami","Navami","Dashami",
    "Ekadashi","Dwadashi","Trayodashi","Chaturdashi","Purnima",
    "Pratipada","Dwitiya","Tritiya","Chaturthi","Panchami",
    "Shashthi","Saptami","Ashtami","Navami","Dashami",
    "Ekadashi","Dwadashi","Trayodashi","Chaturdashi","Amavasya"
]

# ── Helpers ────────────────────────────────────────────────────────
def julian_day(y, m, d, h=0.5):
    """Julian Day Number for a given date and fractional hour (UT)."""
    if m <= 2:
        y -= 1; m += 12
    A = y // 100
    B = 2 - A + A // 4
    return int(365.25*(y+4716)) + int(30.6001*(m+1)) + d + h/24 + B - 1524.5

def daily_motion(revolutions):
    return revolutions * 360.0 / CIVIL_DAYS

def mean_longitude(dm, days):
    return (dm * days) % 360

def manda_correction(mean_lon, mandocca, max_eq):
    """Equation of centre (manda phala)."""
    anomaly = math.radians((mean_lon - mandocca) % 360)
    return max_eq * math.sin(anomaly)

# ── Main calculation ───────────────────────────────────────────────
def calculate_tithi(year, month, day, hour=0.5):
    # Days since Kali Yuga epoch (Feb 18, 3102 BCE proleptic Julian)
    kali_jd = julian_day(-3101, 2, 18)
    today_jd = julian_day(year, month, day, hour)
    days = today_jd - kali_jd

    # Daily motions
    sun_dm    = daily_motion(REV_SUN)
    moon_dm   = daily_motion(REV_MOON)
    apogee_dm = daily_motion(REV_APOGEE)
    node_dm   = daily_motion(REV_NODE)

    # Mean longitudes
    sun_mean   = mean_longitude(sun_dm,   days)
    moon_mean  = mean_longitude(moon_dm,  days)
    apogee     = mean_longitude(apogee_dm, days)
    node       = (-node_dm * days) % 360  # retrograde

    # True longitudes (mean + manda correction)
    sun_eq   = manda_correction(sun_mean,  SUN_APOGEE, SUN_MANDA_MAX)
    moon_eq  = manda_correction(moon_mean, apogee,     MOON_MANDA_MAX)
    sun_true  = (sun_mean  + sun_eq)  % 360
    moon_true = (moon_mean + moon_eq) % 360

    # Tithi
    diff   = (moon_true - sun_true) % 360
    t_num  = diff / 12
    t_idx  = int(t_num)
    frac   = t_num - t_idx

    paksha = "Shukla" if t_idx < 15 else "Krishna"
    name   = TITHI_NAMES[t_idx]

    return {
        "date": f"{year}-{month:02d}-{day:02d} {hour:05.2f}h UT",
        "days_since_kali": days,
        "mean_sun": sun_mean,
        "mean_moon": moon_mean,
        "moon_apogee": apogee,
        "moon_node": node,
        "sun_equation": sun_eq,
        "moon_equation": moon_eq,
        "true_sun": sun_true,
        "true_moon": moon_true,
        "moon_sun_diff": diff,
        "tithi_number": t_idx + 1,
        "tithi_name": name,
        "paksha": paksha,
        "fraction_completed_pct": frac * 100,
    }

def print_result(r):
    print(f"\n{'='*45}")
    print(f"  Surya Siddhanta Tithi Calculator")
    print(f"  Date : {r['date']} (UTC)")
    print(f"{'='*45}")
    print(f"  Days since Kali Yuga : {r['days_since_kali']:,.0f}")
    print(f"  Mean Sun             : {r['mean_sun']:.4f}°")
    print(f"  Mean Moon            : {r['mean_moon']:.4f}°")
    print(f"  Moon Apogee          : {r['moon_apogee']:.4f}°")
    print(f"  Sun  equation        : {r['sun_equation']:+.4f}°")
    print(f"  Moon equation        : {r['moon_equation']:+.4f}°")
    print(f"  True Sun             : {r['true_sun']:.4f}°")
    print(f"  True Moon            : {r['true_moon']:.4f}°")
    print(f"  Moon − Sun           : {r['moon_sun_diff']:.4f}°")
    print(f"{'─'*45}")
    print(f"  TITHI  : {r['paksha']} {r['tithi_name']} ({r['tithi_number']}/30)")
    print(f"  Done   : {r['fraction_completed_pct']:.1f}%")
    print(f"{'='*45}\n")

# ── Modern (Drik) calculation using ephem ─────────────────────────
def calculate_tithi_modern(year, month, day, hour=0.5):
    """Compute tithi using modern planetary ephemeris (matches Drik panchanga)."""
    # Build ephem date string
    dt = datetime(year, month, day, int(hour), int((hour % 1) * 60), tzinfo=timezone.utc)
    d  = ephem.Date(dt.strftime("%Y/%m/%d %H:%M:%S"))

    sun  = ephem.Sun(d)
    moon = ephem.Moon(d)

    sun_lon  = math.degrees(float(sun.hlong))   % 360
    moon_lon = math.degrees(float(moon.hlong))  % 360

    diff  = (moon_lon - sun_lon) % 360
    t_num = diff / 12
    t_idx = int(t_num)
    frac  = t_num - t_idx

    paksha = "Shukla" if t_idx < 15 else "Krishna"
    name   = TITHI_NAMES[t_idx]

    return {
        "method": "Modern (Drik/Ephem)",
        "true_sun":  sun_lon,
        "true_moon": moon_lon,
        "moon_sun_diff": diff,
        "tithi_number": t_idx + 1,
        "tithi_name": name,
        "paksha": paksha,
        "fraction_completed_pct": frac * 100,
    }

def print_comparison(r_ss, r_mod):
    date_str = r_ss["date"]
    print(f"\n{'='*55}")
    print(f"  TITHI COMPARISON — {date_str}")
    print(f"{'='*55}")
    print(f"  {'':25} {'Surya Siddhanta':>14}  {'Modern (Drik)':>13}")
    print(f"  {'─'*53}")
    print(f"  {'True Sun':25} {r_ss['true_sun']:>14.4f}°  {r_mod['true_sun']:>12.4f}°")
    print(f"  {'True Moon':25} {r_ss['true_moon']:>14.4f}°  {r_mod['true_moon']:>12.4f}°")
    print(f"  {'Moon − Sun':25} {r_ss['moon_sun_diff']:>14.4f}°  {r_mod['moon_sun_diff']:>12.4f}°")
    print(f"  {'─'*53}")
    print(f"  {'Paksha':25} {r_ss['paksha']:>14}  {r_mod['paksha']:>13}")
    print(f"  {'Tithi':25} {r_ss['tithi_name']:>14}  {r_mod['tithi_name']:>13}")
    print(f"  {'Number':25} {r_ss['tithi_number']:>14}  {r_mod['tithi_number']:>13}")
    print(f"  {'Completed':25} {r_ss['fraction_completed_pct']:>13.1f}%  {r_mod['fraction_completed_pct']:>12.1f}%")
    diff_deg = abs(r_ss['moon_sun_diff'] - r_mod['moon_sun_diff'])
    print(f"  {'─'*53}")
    print(f"  SS error vs Modern : {diff_deg:.2f}° ({diff_deg/12:.1f} tithis)")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        y, m, d = map(int, sys.argv[1].split("-"))
        h = 0.5  # noon for a specific date
    else:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        y, m, d, h = now.year, now.month, now.day, now.hour + now.minute/60 + now.second/3600

    r_ss  = calculate_tithi(y, m, d, h)
    r_mod = calculate_tithi_modern(y, m, d, h)
    print_result(r_ss)
    print_comparison(r_ss, r_mod)