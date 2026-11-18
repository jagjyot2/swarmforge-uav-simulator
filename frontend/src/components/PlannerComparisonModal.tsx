import React, { useState } from "react";
import { PlannerComparisonResult, Vec3 } from "../types/simulation";

interface PlannerComparisonModalProps {
  open: boolean;
  onClose: () => void;
  onOverlayResult: (result: PlannerComparisonResult | null) => void;
}

export const PlannerComparisonModal: React.FC<PlannerComparisonModalProps> = ({
  open,
  onClose,
  onOverlayResult,
}) => {
  const [start, setStart] = useState<Vec3>({ x: -62, y: -52, z: 18 });
  const [goal, setGoal] = useState<Vec3>({ x: 58, y: 48, z: 24 });
  const [result, setResult] = useState<PlannerComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);

  if (!open) return null;

  const runBenchmark = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/planners/compare?use_mission_obstacles=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start, goal, clearance_m: 2.5 }),
      });
      const data: PlannerComparisonResult = await res.json();
      setResult(data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(2, 6, 23, 0.8)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 50,
      }}
    >
      <div
        style={{
          width: "min(860px, 94vw)",
          background: "#0b1325",
          border: "1px solid rgba(56, 189, 248, 0.45)",
          borderRadius: 12,
          padding: 18,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ margin: 0 }}>3D Path Planner Benchmark: A* vs RRT* vs APF</h3>
          <button onClick={onClose}>Close</button>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(6, 1fr)",
            gap: 8,
            marginTop: 12,
          }}
        >
          {(["x", "y", "z"] as const).map((axis) => (
            <label key={`s-${axis}`} style={{ fontSize: 11, color: "#94a3b8" }}>
              Start {axis.toUpperCase()}
              <input
                type="number"
                value={start[axis]}
                onChange={(e) => setStart({ ...start, [axis]: Number(e.target.value) })}
                style={{ width: "100%", marginTop: 4 }}
              />
            </label>
          ))}
          {(["x", "y", "z"] as const).map((axis) => (
            <label key={`g-${axis}`} style={{ fontSize: 11, color: "#94a3b8" }}>
              Goal {axis.toUpperCase()}
              <input
                type="number"
                value={goal[axis]}
                onChange={(e) => setGoal({ ...goal, [axis]: Number(e.target.value) })}
                style={{ width: "100%", marginTop: 4 }}
              />
            </label>
          ))}
        </div>

        <div className="sf-btn-row" style={{ marginTop: 12 }}>
          <button className="primary" onClick={runBenchmark} disabled={loading}>
            {loading ? "Benchmarking..." : "Run 3D Planner Benchmark"}
          </button>
          {result && (
            <button
              onClick={() => {
                onOverlayResult(result);
                onClose();
              }}
            >
              Overlay Paths in 3D Viewport
            </button>
          )}
        </div>

        {result && (
          <table style={{ width: "100%", marginTop: 14, borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ color: "#38bdf8", textAlign: "left" }}>
                <th>Algorithm</th>
                <th>Status</th>
                <th>Length (m)</th>
                <th>Compute (ms)</th>
                <th>Smoothness (rad)</th>
                <th>Min Clearance (m)</th>
                <th>Energy (Wh)</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((m) => (
                <tr key={m.algorithm} style={{ borderTop: "1px solid rgba(148,163,184,0.2)" }}>
                  <td style={{ padding: "8px 4px", fontWeight: 700 }}>{m.algorithm.toUpperCase()}</td>
                  <td>{m.success ? "PASS" : "FAIL"}</td>
                  <td>{m.path_length_m.toFixed(2)}</td>
                  <td>{m.compute_time_ms.toFixed(2)}</td>
                  <td>{m.smoothness_rad.toFixed(3)}</td>
                  <td>{m.min_obstacle_clearance_m.toFixed(2)}</td>
                  <td>{m.estimated_energy_wh.toFixed(3)}</td>
                  <td>{m.algorithm === result.recommended_algorithm ? "★ RECOMMENDED" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
