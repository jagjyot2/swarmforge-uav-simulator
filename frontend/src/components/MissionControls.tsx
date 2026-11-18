import React, { useRef, useState } from "react";
import { MissionSummary, PlannerType, TelemetryFrame } from "../types/simulation";

interface MissionControlsProps {
  missions: MissionSummary[];
  activeMissionId: string;
  onFrameUpdate: (frame: TelemetryFrame) => void;
  onToast: (msg: string) => void;
}

export const MissionControls: React.FC<MissionControlsProps> = ({
  missions,
  activeMissionId,
  onFrameUpdate,
  onToast,
}) => {
  const [selectedMission, setSelectedMission] = useState(activeMissionId);
  const [vx, setVx] = useState(2.4);
  const [vy, setVy] = useState(-1.1);
  const [turb, setTurb] = useState(0.75);
  const [commAction, setCommAction] = useState<"RTL" | "HOLD">("RTL");
  const fileRef = useRef<HTMLInputElement | null>(null);

  const post = async (url: string, body: unknown = {}) => {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(JSON.stringify(data));
    return data;
  };

  const switchPlanner = async (algorithm: PlannerType) => {
    const frame = await post("/api/simulation/planner", { algorithm });
    onFrameUpdate(frame);
    onToast(`Swarm replanned with ${algorithm.toUpperCase()}`);
  };

  return (
    <aside className="sf-sidebar">
      <div className="sf-card">
        <h4>Simulation Transport</h4>
        <div className="sf-btn-row">
          <button className="primary" onClick={async () => onFrameUpdate(await post("/api/simulation/start"))}>
            Start
          </button>
          <button onClick={async () => onFrameUpdate(await post("/api/simulation/pause"))}>
            Pause
          </button>
          <button onClick={async () => onFrameUpdate(await post("/api/simulation/step", { steps: 20 }))}>
            Step +20
          </button>
          <button onClick={async () => onFrameUpdate(await post("/api/simulation/reset"))}>
            Reset
          </button>
        </div>
      </div>

      <div className="sf-card">
        <h4>Mission Preset & JSON I/O</h4>
        <select
          value={selectedMission || activeMissionId}
          onChange={(e) => setSelectedMission(e.target.value)}
          style={{ width: "100%", marginBottom: 8 }}
        >
          {missions.map((m) => (
            <option key={m.mission_id} value={m.mission_id}>
              {m.name} ({m.uav_count} UAVs)
            </option>
          ))}
        </select>
        <div className="sf-btn-row">
          <button
            className="primary"
            onClick={async () => {
              const res = await post("/api/missions/load", {
                mission_id: selectedMission || activeMissionId,
              });
              onFrameUpdate(res.telemetry);
              onToast(`Loaded ${res.mission.name}`);
            }}
          >
            Load
          </button>
          <button onClick={() => (window.location.href = "/api/missions/export")}>
            Export JSON
          </button>
          <button onClick={() => fileRef.current?.click()}>Import JSON</button>
          <input
            ref={fileRef}
            type="file"
            accept=".json"
            style={{ display: "none" }}
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              const json = JSON.parse(await file.text());
              const res = await post("/api/missions/load", json);
              onFrameUpdate(res.telemetry);
              onToast(`Imported ${res.mission.name}`);
            }}
          />
        </div>
      </div>

      <div className="sf-card">
        <h4>Global Planner Override</h4>
        <div className="sf-btn-row">
          <button onClick={() => switchPlanner("astar")}>A*</button>
          <button onClick={() => switchPlanner("rrt_star")}>RRT*</button>
          <button onClick={() => switchPlanner("apf")}>APF</button>
        </div>
      </div>

      <div className="sf-card">
        <h4>Wind & Dryden Turbulence</h4>
        <label style={{ fontSize: 11, display: "block", marginBottom: 6 }}>
          Vx (East): {vx.toFixed(1)} m/s
          <input
            type="range"
            min={-10}
            max={10}
            step={0.5}
            value={vx}
            onChange={(e) => setVx(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </label>
        <label style={{ fontSize: 11, display: "block", marginBottom: 6 }}>
          Vy (North): {vy.toFixed(1)} m/s
          <input
            type="range"
            min={-10}
            max={10}
            step={0.5}
            value={vy}
            onChange={(e) => setVy(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </label>
        <label style={{ fontSize: 11, display: "block", marginBottom: 8 }}>
          Turbulence σ: {turb.toFixed(2)} m/s
          <input
            type="range"
            min={0}
            max={4}
            step={0.1}
            value={turb}
            onChange={(e) => setTurb(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </label>
        <button
          style={{ width: "100%" }}
          onClick={async () => {
            const frame = await post("/api/simulation/wind", {
              steady_vx_mps: vx,
              steady_vy_mps: vy,
              steady_vz_mps: 0,
              turbulence_intensity: turb,
              gust_frequency_hz: 0.35,
              seed: 42,
            });
            onFrameUpdate(frame);
            onToast("Wind field updated");
          }}
        >
          Apply Wind Field
        </button>
      </div>

      <div className="sf-card">
        <h4>Comm-Loss & Emergency Failsafes</h4>
        <select
          value={commAction}
          onChange={(e) => setCommAction(e.target.value as "RTL" | "HOLD")}
          style={{ width: "100%", marginBottom: 8 }}
        >
          <option value="RTL">Link Drop → Autonomous RTL</option>
          <option value="HOLD">Link Drop → Autonomous HOLD</option>
        </select>
        <div className="sf-btn-row" style={{ marginBottom: 8 }}>
          <button
            className="warn"
            onClick={async () =>
              onFrameUpdate(
                await post("/api/simulation/comm-loss", { active: true, action: commAction })
              )
            }
          >
            Drop Link
          </button>
          <button
            onClick={async () =>
              onFrameUpdate(await post("/api/simulation/comm-loss", { active: false }))
            }
          >
            Restore Link
          </button>
        </div>
        <div className="sf-btn-row">
          <button
            className="danger"
            onClick={async () =>
              onFrameUpdate(await post("/api/simulation/emergency", { command: "RTL" }))
            }
          >
            Emergency RTL
          </button>
          <button
            className="danger"
            onClick={async () =>
              onFrameUpdate(await post("/api/simulation/emergency", { command: "LAND" }))
            }
          >
            Emergency Land
          </button>
          <button
            onClick={async () =>
              onFrameUpdate(await post("/api/simulation/emergency", { command: "CLEAR" }))
            }
          >
            Clear
          </button>
        </div>
      </div>
    </aside>
  );
};
