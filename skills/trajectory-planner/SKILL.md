---
name: trajectory-planner
description: Produce frame-indexed entity and camera trajectories from a validated SceneContract.
---

# Trajectory Planner Skill

The planner receives only a validated `SceneContract`. It emits `TrajectoryPlan` states, motion primitives, attachment transitions, camera shots, and event observability records. Positions and camera shots use the contract FPS and a one-based inclusive frame range.

Every required event must be covered by a shot with a target-visible predicate. State frames are strictly increasing, attachment transitions are explicit, and discontinuous motion is rejected before Blender execution. Runtime repair routes are `trajectory_repair` and `camera_repair`; event semantics are not silently rewritten.
