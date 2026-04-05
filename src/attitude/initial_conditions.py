"""
initial_conditions.py — Convert launch-site parameters to 6DOF ECI initial state.

The ECI frame is Earth-Centered Inertial with the x-axis pointing toward
the vernal equinox.  For simulation purposes we set GAST = 0 at t = 0
(i.e., the ECI x-axis coincides with the Greenwich meridian at launch time).
This is an idealised choice; a higher-fidelity implementation would account
for the actual sidereal time.

Outputs
-------
rx, ry, rz      ECI position [m]
vx_eci, vy_eci, vz_eci  ECI velocity [m/s]  (Earth-rotation contribution)
qw, qx, qy, qz  body-to-ECI quaternion  (body-x = radial-up at launch)
"""

from __future__ import annotations

import math

from src.attitude.quaternion import Quat, Vec3, quat_from_two_vectors, vec3_normalize, vec3_cross

OMEGA_EARTH = 7.2921150e-5  # Earth rotation rate [rad/s]
R_EARTH     = 6_371_000.0   # mean Earth radius [m]


def launch_site_eci(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Return (r_eci, v_eci) for a launch site at rest on the rotating Earth.

    Parameters
    ----------
    latitude_deg  : geodetic latitude [deg]
    longitude_deg : longitude [deg]
    altitude_m    : altitude above mean sea level [m]

    Returns
    -------
    r_eci : (rx, ry, rz) [m]
    v_eci : (vx, vy, vz) [m/s]  — due to Earth rotation
    """
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    r   = R_EARTH + altitude_m

    rx = r * math.cos(lat) * math.cos(lon)
    ry = r * math.cos(lat) * math.sin(lon)
    rz = r * math.sin(lat)

    # v = ω_E × r,  ω_E = [0, 0, ω_E]
    vx_eci = -OMEGA_EARTH * ry
    vy_eci =  OMEGA_EARTH * rx
    vz_eci = 0.0

    return (rx, ry, rz), (vx_eci, vy_eci, vz_eci)


def launch_attitude(
    rx: float, ry: float, rz: float,
    latitude_deg: float,
    longitude_deg: float,
    azimuth_deg: float,
) -> Quat:
    """
    Return the initial body-to-ECI quaternion.

    At launch the vehicle stands vertical, so:
        body-x  =  radial unit vector (pointing straight up)
        body-z  =  aligned with the launch plane normal (right-hand rule)
        body-y  =  body-z × body-x  (completes right-hand frame)

    The azimuth is used to orient the pitch plane (body-y/body-x) correctly
    in the launch-plane for pitch-programme purposes.

    Implementation note
    -------------------
    We build the rotation matrix from the three body-axis directions expressed
    in ECI and then convert it to a quaternion.  The vertical axis (body-x) is
    always r_hat; the azimuth defines the horizontal forward direction from
    which we derive body-y and body-z.
    """
    r_mag = math.sqrt(rx * rx + ry * ry + rz * rz)
    if r_mag < 1.0:
        return (1.0, 0.0, 0.0, 0.0)

    # Radial (up) direction — body-x
    r_hat: Vec3 = (rx / r_mag, ry / r_mag, rz / r_mag)

    # Local East unit vector in ECI
    lon = math.radians(longitude_deg)
    east: Vec3 = (-math.sin(lon), math.cos(lon), 0.0)

    # Local North unit vector in ECI
    lat = math.radians(latitude_deg)
    # North = (−sin(lat)cos(lon), −sin(lat)sin(lon), cos(lat))
    north: Vec3 = (
        -math.sin(lat) * math.cos(lon),
        -math.sin(lat) * math.sin(lon),
         math.cos(lat),
    )

    # Horizontal launch direction (in body pitch plane)
    az = math.radians(azimuth_deg)
    # forward_h = cos(az)*north + sin(az)*east
    fwd_x = math.cos(az) * north[0] + math.sin(az) * east[0]
    fwd_y = math.cos(az) * north[1] + math.sin(az) * east[1]
    fwd_z = math.cos(az) * north[2] + math.sin(az) * east[2]
    forward_h: Vec3 = vec3_normalize((fwd_x, fwd_y, fwd_z))

    # body-z = r_hat × forward_h  (out-of-plane to the right)
    body_z = vec3_normalize(vec3_cross(r_hat, forward_h))

    # body-y = body-z × body-x  (completes right-hand frame; in pitch plane)
    body_y = vec3_normalize(vec3_cross(body_z, r_hat))

    # Build rotation matrix R (columns = body-x, body-y, body-z expressed in ECI)
    # This is the rotation that takes ECI unit vectors to body-frame unit vectors,
    # so R^T transforms from body to ECI → that is our quaternion.
    # Column-major: R[:,0]=body-x_in_ECI = r_hat, R[:,1]=body-y_in_ECI, R[:,2]=body-z_in_ECI
    # Actually: the body-to-ECI rotation matrix has body axes as its columns:
    #   R_b2e = [body-x_eci | body-y_eci | body-z_eci]
    r00, r10, r20 = r_hat        # body-x expressed in ECI
    r01, r11, r21 = body_y       # body-y expressed in ECI
    r02, r12, r22 = body_z       # body-z expressed in ECI

    # Rotation matrix to quaternion (Shepperd's method)
    trace = r00 + r11 + r22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (r21 - r12) * s
        qy = (r02 - r20) * s
        qz = (r10 - r01) * s
    elif r00 > r11 and r00 > r22:
        s = 2.0 * math.sqrt(1.0 + r00 - r11 - r22)
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * math.sqrt(1.0 + r11 - r00 - r22)
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r22 - r00 - r11)
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    return (qw/n, qx/n, qy/n, qz/n)
