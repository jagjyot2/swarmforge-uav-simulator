import React, { useEffect, useState } from "react";
import { MissionControls } from "./components/MissionControls";
import { PlannerComparisonModal } from "./components/PlannerComparisonModal";
import { SwarmViewport3D } from "./components/SwarmViewport3D";
import { TelemetryPanel } from "./components/TelemetryPanel";
import {
  MissionSummary,
  PlannerComparisonResult,
  TelemetryFrame,
} from "./types/simulation";

export const App: React.FC = () => {
  const [frame, setFrame] = useState<TelemetryFrame | null>(null);
  const [missions, setMissions] = useState<MissionSummary[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareOverlay, setCompareOverlay] = useState<PlannerComparisonResult | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const notify = (msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), 3500);
  };

  useEffect(() => {
    fetch("/api/missions")
      .then((r) => r.json())
      .then((d) => setMissions(d.missions || []))
      .catch(() => {});

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/telemetry`);
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onmessage = (evt) => {
      try {
        setFrame(JSON.parse(evt.data));
      } catch {
        // ignore malformed telemetry frame
      }
    };
    return () => ws.close();
  }, []);

  const testHardwareLock = async () => {
    const res = await fetch("/api/hardware/actuate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: "ARM_MOTORS", target_uav_id: "uav-01" }),
    });
    const body = await res.json();
    notify(`HTTP ${res.status} BLOCKED: ${body?.detail?.message || "Hardware actuation locked"}`);
  };

  return (
    <div className="sf-shell">
      <header className="sf-topbar">
        <div style={{ fontWeight: 700, letterSpacing: "0.06em", fontSize: 14 }}>
          SWARMFORGE // 3D MULTI-UAV DIGITAL TWIN
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span
            style={{
              padding: "4px 10px",
              borderRadius: 999,
              background: "rgba(16, 185, 129, 0.14)",
              border: "1px solid rgba(16, 185, 129, 0.5)",
              color: "#6ee7b7",
              fontSize: 11,
              fontWeight: 700,
            }}
          >
            HARDWARE ACTUATION LOCKED = {String(frame?.hardware_actuation_locked ?? true).toUpperCase()}
          </span>
          <button onClick={testHardwareLock}>Test Lock Guard (403)</button>
          <span style={{ fontFamily: "monospace", fontSize: 12 }}>
            T+{(frame?.sim_time_s ?? 0).toFixed(2)}s | {frame?.status ?? "CONNECTING"}
          </span>
          <span
            style={{
              fontFamily: "monospace",
              fontSize: 11,
              color: wsConnected ? "#38bdf8" : "#fca5a5",
            }}
          >
            {wsConnected ? "WS: LIVE" : "WS: OFFLINE"}
          </span>
        </div>
        <button className="primary" onClick={() => setCompareOpen(true)}>
          Compare A* / RRT* / APF
        </button>
      </header>

      <div className="sf-main">
        <MissionControls
          missions={missions}
          activeMissionId={frame?.mission_id ?? "urban_corridor"}
          onFrameUpdate={setFrame}
          onToast={notify}
        />

        <main style={{ position: "relative", width: "100%", height: "100%" }}>
          <SwarmViewport3D frame={frame} compareOverlay={compareOverlay} />
          {compareOverlay && (
            <button
              style={{ position: "absolute", bottom: 12, left: 12 }}
              onClick={() => setCompareOverlay(null)}
            >
              Clear Benchmark Overlay
            </button>
          )}
        </main>

        <TelemetryPanel frame={frame} />
      </div>

      <PlannerComparisonModal
        open={compareOpen}
        onClose={() => setCompareOpen(false)}
        onOverlayResult={setCompareOverlay}
      />

      {toast && (
        <div
          style={{
            position: "fixed",
            bottom: 18,
            right: 18,
            background: "#0f172a",
            border: "1px solid #38bdf8",
            padding: "10px 14px",
            borderRadius: 8,
            fontSize: 12,
          }}
        >
          {toast}
        </div>
      )}
    </div>
  );
};

export default App;
