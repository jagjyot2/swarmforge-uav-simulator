import React from "react";
import { TelemetryFrame } from "../types/simulation";

interface TelemetryPanelProps {
  frame: TelemetryFrame | null;
}

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({ frame }) => {
  if (!frame) {
    return (
      <aside className="sf-sidebar right">
        <div className="sf-card">
          <h4>Swarm Live Telemetry</h4>
          <p style={{ fontSize: 12, color: "#94a3b8" }}>Connecting to telemetry stream...</p>
        </div>
      </aside>
    );
  }

  return (
    <aside className="sf-sidebar right">
      <div className="sf-card">
        <h4>Swarm Live Telemetry ({frame.uavs.length} UAVs)</h4>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {frame.uavs.map((u) => {
            const socColor =
              u.battery_soc_pct > 50 ? "#10b981" : u.battery_soc_pct > 25 ? "#f59e0b" : "#ef4444";
            return (
              <div
                key={u.uav_id}
                style={{
                  background: "rgba(15, 23, 42, 0.9)",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderLeft: `3px solid ${u.color}`,
                  borderRadius: 8,
                  padding: 9,
                  fontSize: 11,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", fontWeight: 700 }}>
                  <span>
                    {u.callsign} [{u.planner_type.toUpperCase()}]
                  </span>
                  <span style={{ fontFamily: "monospace", color: "#38bdf8" }}>{u.mode}</span>
                </div>
                <div
                  style={{
                    height: 5,
                    borderRadius: 99,
                    background: "rgba(148, 163, 184, 0.2)",
                    margin: "5px 0",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      width: `${u.battery_soc_pct}%`,
                      height: "100%",
                      background: socColor,
                    }}
                  />
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "4px 8px",
                    fontFamily: "monospace",
                    fontSize: 10.5,
                    color: "#cbd5e1",
                  }}
                >
                  <span>SoC: {u.battery_soc_pct.toFixed(1)}%</span>
                  <span>Volt: {u.voltage_v.toFixed(2)} V</span>
                  <span>Power: {u.power_w.toFixed(0)} W</span>
                  <span>Energy: {u.energy_consumed_wh.toFixed(2)} Wh</span>
                  <span>GndSpd: {u.speed_mps.toFixed(1)} m/s</span>
                  <span>AirSpd: {u.airspeed_mps.toFixed(1)} m/s</span>
                  <span>
                    Pos: {u.position.x.toFixed(1)},{u.position.y.toFixed(1)},
                    {u.position.z.toFixed(1)}
                  </span>
                  <span>RPM: {u.rotor_rpm.toFixed(0)}</span>
                </div>
                {u.alerts.length > 0 && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 5 }}>
                    {u.alerts.map((al) => (
                      <span
                        key={al}
                        style={{
                          fontSize: 9.5,
                          padding: "1px 5px",
                          borderRadius: 4,
                          background: "rgba(239, 68, 68, 0.2)",
                          border: "1px solid rgba(239, 68, 68, 0.45)",
                          color: "#fecaca",
                          fontFamily: "monospace",
                        }}
                      >
                        {al}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="sf-card">
        <h4>Safety & Event Log</h4>
        <div
          style={{
            fontFamily: "monospace",
            fontSize: 10.5,
            maxHeight: 170,
            overflowY: "auto",
            color: "#94a3b8",
            display: "flex",
            flexDirection: "column",
            gap: 3,
          }}
        >
          {frame.events
            .slice()
            .reverse()
            .map((ev, i) => (
              <div key={i}>{ev}</div>
            ))}
        </div>
      </div>
    </aside>
  );
};
