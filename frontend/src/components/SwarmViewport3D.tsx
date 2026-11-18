import React, { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import {
  Geofence,
  Obstacle3D,
  PlannerComparisonResult,
  TelemetryFrame,
  UAVTelemetry,
  Vec3,
} from "../types/simulation";

const PLANNER_COLORS: Record<string, string> = {
  astar: "#38bdf8",
  rrt_star: "#a855f7",
  apf: "#f59e0b",
  direct: "#10b981",
};

/** Convert ENU coordinates (x=East, y=North, z=Up) to Three.js Y-up space. */
function enuToThree(p: Vec3): [number, number, number] {
  return [p.x, p.z, -p.y];
}

const ROTOR_OFFSETS: [number, number, number][] = [
  [1.55, 0.32, 1.55],
  [-1.55, 0.32, 1.55],
  [1.55, 0.32, -1.55],
  [-1.55, 0.32, -1.55],
];

const QuadcopterMesh: React.FC<{ uav: UAVTelemetry }> = ({ uav }) => {
  const rotorRefs = useRef<(THREE.Mesh | null)[]>([]);

  useFrame((_, delta) => {
    const rate = (uav.rotor_rpm / 5200) * 32.0 * delta;
    rotorRefs.current.forEach((mesh, idx) => {
      if (mesh) {
        mesh.rotation.y += (idx % 2 === 0 ? 1 : -1) * rate;
      }
    });
  });

  const pos = enuToThree(uav.position);
  const rotY = (uav.attitude_deg.z * Math.PI) / 180 + Math.PI / 2;
  const rotX = -(uav.attitude_deg.y * Math.PI) / 180;
  const rotZ = -(uav.attitude_deg.x * Math.PI) / 180;

  return (
    <group>
      <group position={pos} rotation={[rotX, rotY, rotZ, "YXZ"]}>
        <mesh>
          <boxGeometry args={[1.8, 0.65, 2.2]} />
          <meshStandardMaterial color={uav.color} metalness={0.4} roughness={0.35} />
        </mesh>
        <mesh position={[0, 0.1, -1.4]} rotation={[Math.PI / 2, 0, 0]}>
          <coneGeometry args={[0.42, 0.9, 8]} />
          <meshBasicMaterial color="#fef08a" />
        </mesh>
        <mesh rotation={[0, Math.PI / 4, 0]}>
          <boxGeometry args={[4.4, 0.18, 0.22]} />
          <meshStandardMaterial color="#cbd5e1" />
        </mesh>
        <mesh rotation={[0, -Math.PI / 4, 0]}>
          <boxGeometry args={[4.4, 0.18, 0.22]} />
          <meshStandardMaterial color="#cbd5e1" />
        </mesh>
        {ROTOR_OFFSETS.map(([ox, oy, oz], idx) => (
          <group key={idx} position={[ox, oy, oz]}>
            <mesh
              ref={(el) => {
                rotorRefs.current[idx] = el;
              }}
            >
              <boxGeometry args={[1.5, 0.06, 0.18]} />
              <meshBasicMaterial color="#38bdf8" transparent opacity={0.85} />
            </mesh>
            <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, -0.02, 0]}>
              <ringGeometry args={[0.65, 0.78, 16]} />
              <meshBasicMaterial
                color={uav.color}
                side={THREE.DoubleSide}
                transparent
                opacity={0.45}
              />
            </mesh>
          </group>
        ))}
      </group>
      <Line
        points={[pos, [pos[0], 0, pos[2]]]}
        color={uav.color}
        transparent
        opacity={0.35}
        lineWidth={1}
      />
    </group>
  );
};

const ObstacleMesh: React.FC<{ obs: Obstacle3D }> = ({ obs }) => {
  const yCenter = obs.base_z + obs.height_z * 0.5;
  if (obs.kind === "cylinder") {
    return (
      <mesh position={[obs.x, yCenter, -obs.y]}>
        <cylinderGeometry args={[obs.radius, obs.radius, obs.height_z, 24]} />
        <meshStandardMaterial color={obs.color || "#475569"} roughness={0.7} />
      </mesh>
    );
  }
  return (
    <mesh position={[obs.x, yCenter, -obs.y]}>
      <boxGeometry args={[obs.width_x, obs.height_z, obs.depth_y]} />
      <meshStandardMaterial color={obs.color || "#334155"} roughness={0.72} />
    </mesh>
  );
};

const GeofenceCylinder: React.FC<{ geofence: Geofence; breached: boolean; warning: boolean }> = ({
  geofence,
  breached,
  warning,
}) => {
  if (!geofence.enabled) return null;
  const color = breached ? "#ef4444" : warning ? "#f59e0b" : "#10b981";
  return (
    <group position={[geofence.center_x, 0, -geofence.center_y]}>
      <mesh position={[0, geofence.max_altitude_m * 0.5, 0]}>
        <cylinderGeometry
          args={[geofence.radius_m, geofence.radius_m, geofence.max_altitude_m, 48, 1, true]}
        />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.08}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <mesh position={[0, geofence.max_altitude_m, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[geofence.radius_m - 0.5, geofence.radius_m + 0.5, 64]} />
        <meshBasicMaterial color={color} side={THREE.DoubleSide} transparent opacity={0.45} />
      </mesh>
    </group>
  );
};

interface SwarmViewport3DProps {
  frame: TelemetryFrame | null;
  compareOverlay: PlannerComparisonResult | null;
}

export const SwarmViewport3D: React.FC<SwarmViewport3DProps> = ({ frame, compareOverlay }) => {
  const anyWarning = useMemo(
    () => Boolean(frame?.uavs.some((u) => u.geofence_warning)),
    [frame]
  );
  const anyBreached = useMemo(
    () => Boolean(frame?.uavs.some((u) => u.geofence_breached)),
    [frame]
  );

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas camera={{ position: [140, 125, 140], fov: 50, near: 0.5, far: 1200 }}>
        <color attach="background" args={["#070b14"]} />
        <fogExp2 attach="fog" args={["#070b14", 0.0018]} />
        <ambientLight intensity={0.65} color="#93c5fd" />
        <directionalLight position={[120, 180, 90]} intensity={1.15} />
        <gridHelper args={[320, 32, "#1e3a5f", "#0f1f38"]} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} />

        {frame?.obstacles.map((obs) => (
          <ObstacleMesh key={obs.id} obs={obs} />
        ))}

        {frame?.geofence && (
          <GeofenceCylinder
            geofence={frame.geofence}
            breached={anyBreached}
            warning={anyWarning}
          />
        )}

        {frame?.uavs.map((uav) => {
          const lineColor = PLANNER_COLORS[uav.planner_type] || uav.color;
          return (
            <React.Fragment key={uav.uav_id}>
              <QuadcopterMesh uav={uav} />
              {uav.planned_path.length >= 2 && (
                <Line
                  points={uav.planned_path.map(enuToThree)}
                  color={lineColor}
                  lineWidth={2}
                />
              )}
              {uav.mission_waypoints.map((wp, idx) => (
                <mesh key={idx} position={enuToThree(wp)}>
                  <octahedronGeometry args={[1.25, 0]} />
                  <meshBasicMaterial color={lineColor} wireframe />
                </mesh>
              ))}
            </React.Fragment>
          );
        })}

        {compareOverlay?.results.map((res) =>
          res.success && res.waypoints.length >= 2 ? (
            <Line
              key={res.algorithm}
              points={res.waypoints.map(enuToThree)}
              color={PLANNER_COLORS[res.algorithm] || "#ffffff"}
              lineWidth={3}
            />
          ) : null
        )}
      </Canvas>
    </div>
  );
};
