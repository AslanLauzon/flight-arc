from __future__ import annotations

from io import BytesIO
import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from src.events.event import Event
from src.orbital.elements import OrbitalElements
from src.propagator.state import SimState

R_EARTH_KM = 6371.0
MU = 3.986004418e14


def plot_trajectory(
    state: SimState,
    events: list[Event],
    elements: OrbitalElements | None = None,
    save_path: str | None = None,
) -> None:
    fig = create_trajectory_figure(state, events, elements)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def render_trajectory_plot_png(
    state: SimState,
    events: list[Event],
    elements: OrbitalElements | None = None,
) -> bytes:
    fig = create_trajectory_figure(state, events, elements)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def create_trajectory_figure(
    state: SimState,
    events: list[Event],
    elements: OrbitalElements | None = None,
) -> plt.Figure:
    df = pd.DataFrame(state.history)
    fired = [event for event in events if event.result is not None]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Nominal Trajectory - Flight Summary", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35, height_ratios=[1, 1, 1.4])

    ax_alt = fig.add_subplot(gs[0, 0])
    ax_spd = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[0, 2])
    ax_mach = fig.add_subplot(gs[1, 0])
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_traj = fig.add_subplot(gs[1, 2])
    ax_orb = fig.add_subplot(gs[2, :])

    t = df["t"]
    altitude_km = df["y"] / 1e3
    downrange_km = df["x"] / 1e3
    speed_km_s = np.hypot(df["vx"], df["vy"]) / 1e3
    q_kpa = df["dynamic_pressure_Pa"] / 1e3
    mach = df["mach"]
    vx_km_s = df["vx"] / 1e3
    vy_km_s = df["vy"] / 1e3

    ax_alt.plot(t, altitude_km, color="steelblue", linewidth=1.5)
    ax_alt.set_xlabel("Time [s]")
    ax_alt.set_ylabel("Altitude [km]")
    ax_alt.set_title("Altitude vs Time")
    _add_event_lines(ax_alt, fired)

    ax_spd.plot(t, speed_km_s, color="seagreen", linewidth=1.5)
    ax_spd.set_xlabel("Time [s]")
    ax_spd.set_ylabel("Speed [km/s]")
    ax_spd.set_title("Speed vs Time")
    _add_event_lines(ax_spd, fired)

    ax_q.plot(t, q_kpa, color="crimson", linewidth=1.5)
    ax_q.set_xlabel("Time [s]")
    ax_q.set_ylabel("Dynamic Pressure [kPa]")
    ax_q.set_title("Dynamic Pressure vs Time")
    _add_event_lines(ax_q, fired)

    ax_mach.plot(t, mach, color="purple", linewidth=1.5)
    ax_mach.set_xlabel("Time [s]")
    ax_mach.set_ylabel("Mach [-]")
    ax_mach.set_title("Mach vs Time")
    _add_event_lines(ax_mach, fired)

    ax_vel.plot(t, vx_km_s, label="vx", color="steelblue", linewidth=1.5)
    ax_vel.plot(t, vy_km_s, label="vy", color="darkorange", linewidth=1.5)
    ax_vel.set_xlabel("Time [s]")
    ax_vel.set_ylabel("Velocity [km/s]")
    ax_vel.set_title("Velocity Components vs Time")
    ax_vel.legend(fontsize=8)
    _add_event_lines(ax_vel, fired)

    ax_traj.plot(downrange_km, altitude_km, color="darkorange", linewidth=1.5)
    ax_traj.set_xlabel("Downrange [km]")
    ax_traj.set_ylabel("Altitude [km]")
    ax_traj.set_title("Trajectory Shape")

    _plot_orbit_panel(ax_orb, df, state, elements)
    _add_event_legend(fig, fired)
    return fig


def _eci_track(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    radius_km = R_EARTH_KM + df["y"].to_numpy() / 1e3
    theta = df["x"].to_numpy() / (R_EARTH_KM * 1e3)
    return radius_km * np.cos(theta), radius_km * np.sin(theta)


def _plot_orbit_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    state: SimState,
    elements: OrbitalElements | None,
) -> None:
    ex, ey = _eci_track(df)
    max_radius_km = np.max(R_EARTH_KM + df["y"].to_numpy() / 1e3)

    earth = Circle((0, 0), R_EARTH_KM, color="#1a6faf", alpha=0.85, zorder=2)
    atmosphere = Circle((0, 0), R_EARTH_KM + 100, color="#87ceeb", alpha=0.15, zorder=1)
    ax.add_patch(earth)
    ax.add_patch(atmosphere)

    ax.plot(ex, ey, color="darkorange", linewidth=1.5, zorder=4, label="Ascent trajectory")
    ax.plot(ex[0], ey[0], "o", color="lime", markersize=6, zorder=5, label="Launch")
    ax.plot(ex[-1], ey[-1], "x", color="red", markersize=8, markeredgewidth=2, zorder=5, label="SECO")

    if elements is not None and elements.perigee_alt_km > -R_EARTH_KM:
        _draw_orbital_ellipse(ax, state, elements)

    limit = max(
        max_radius_km,
        R_EARTH_KM + (elements.apogee_alt_km if elements is not None else 500.0),
    ) + R_EARTH_KM * 0.5

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("ECI x [km]")
    ax.set_ylabel("ECI y [km]")
    ax.set_title("Orbit Plane View")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_facecolor("#0a0a1a")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def _draw_orbital_ellipse(
    ax: plt.Axes,
    state: SimState,
    elements: OrbitalElements,
) -> None:
    earth_radius_m = R_EARTH_KM * 1e3
    theta_seco = state.x / earth_radius_m
    r_seco = earth_radius_m + state.y
    rx = r_seco * math.cos(theta_seco)
    ry = r_seco * math.sin(theta_seco)

    vx_eci = state.vx * (-math.sin(theta_seco)) + state.vy * math.cos(theta_seco)
    vy_eci = state.vx * math.cos(theta_seco) + state.vy * math.sin(theta_seco)

    h = rx * vy_eci - ry * vx_eci
    r = math.hypot(rx, ry)
    ex_vec = vy_eci * h / MU - rx / r
    ey_vec = -vx_eci * h / MU - ry / r
    omega = math.atan2(ey_vec, ex_vec)

    a = elements.semi_major_axis_m
    e = elements.eccentricity
    p = a * (1 - e ** 2)
    nu = np.linspace(0.0, 2.0 * math.pi, 1000)
    radius = p / (1 + e * np.cos(nu))

    xe = radius * np.cos(nu + omega) / 1e3
    ye = radius * np.sin(nu + omega) / 1e3
    mask = radius / 1e3 > R_EARTH_KM

    ax.plot(
        np.where(mask, xe, np.nan),
        np.where(mask, ye, np.nan),
        color="cyan",
        linewidth=1.2,
        linestyle="--",
        zorder=3,
        label=f"Orbit {elements.perigee_alt_km:.0f}x{elements.apogee_alt_km:.0f} km",
    )

    r_p = a * (1 - e)
    r_a = a * (1 + e)
    ax.plot(r_p * math.cos(omega) / 1e3, r_p * math.sin(omega) / 1e3, "v", color="cyan", markersize=6, zorder=5)
    ax.plot(
        r_a * math.cos(omega + math.pi) / 1e3,
        r_a * math.sin(omega + math.pi) / 1e3,
        "^",
        color="cyan",
        markersize=6,
        zorder=5,
    )


def _add_event_lines(ax: plt.Axes, fired: list[Event]) -> None:
    colors = {
        "ignition": "#e67e22",
        "liftoff": "#27ae60",
        "max_q": "#e74c3c",
        "staging": "#8e44ad",
        "fairing_deploy": "#2980b9",
        "seco": "#c0392b",
        "payload_sep": "#16a085",
    }
    for event in fired:
        ax.axvline(
            x=event.result.t_trigger,
            color=colors.get(event.name, "gray"),
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )


def _add_event_legend(fig: plt.Figure, fired: list[Event]) -> None:
    colors = {
        "ignition": "#e67e22",
        "liftoff": "#27ae60",
        "max_q": "#e74c3c",
        "staging": "#8e44ad",
        "fairing_deploy": "#2980b9",
        "seco": "#c0392b",
        "payload_sep": "#16a085",
    }
    handles = [
        plt.Line2D([0], [0], color=colors.get(event.name, "gray"), linestyle="--", linewidth=1.5)
        for event in fired
    ]
    labels = [f"{event.name} t={event.result.t_trigger:.0f}s" for event in fired]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(1, len(fired)),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
