"""
make_world.py — turn a Sionna scene into a Gazebo Harmonic world.

Reads a parquet sample (building_mask + tx_origin), and writes an SDF world:
  * obstacle blocks (occupied mask cells, greedily merged into rectangles),
  * an emissive marker at the emitter pixel,
  * a diff-drive rover spawned at a free start cell.

Pixel<->world mapping (RES = 1.7 m/pixel, 256x256 grid):
    world_x = col * RES
    world_y = (255 - row) * RES
The rf_seeker node MUST use the same mapping (see pixel_of_xy there).

Usage:
    python3 make_world.py --data sample_28702.parquet --out rfsim.world
"""
import argparse
import numpy as np
import pandas as pd

RES = 1.7          # metres per pixel
N = 256
WALL_H = 6.0       # obstacle height (m)
ROVER = "rover"


def load_mask(path):
    row = pd.read_parquet(path).iloc[0]
    mask = np.array(row["building_mask"]).reshape(N, N).astype(np.uint8)
    tx = np.array(row["tx_origin"]).reshape(N, N)
    ty, txx = np.unravel_index(int(np.argmax(tx)), (N, N))
    return mask, (int(ty), int(txx))


def merge_rectangles(grid):
    """Greedy maximal-rectangle decomposition. Returns (r0, c0, h, w) in the
    grid's own cell units (used on the full mask or a decimated one)."""
    H, W = grid.shape
    m = grid.copy()
    rects = []
    for r in range(H):
        c = 0
        while c < W:
            if not m[r, c]:
                c += 1
                continue
            c2 = c
            while c2 < W and m[r, c2]:
                c2 += 1
            w = c2 - c
            h = 1
            while r + h < H and m[r + h, c:c2].all():
                h += 1
            m[r:r + h, c:c2] = 0
            rects.append((r, c, h, w))
            c = c2
    return rects


def free_start(mask, tx_pix, target_dist_px=150.0):
    """Pick a free cell roughly target_dist from the emitter (a solvable but
    non-trivial start)."""
    free = np.argwhere(mask == 0)
    d = np.hypot(free[:, 0] - tx_pix[0], free[:, 1] - tx_pix[1])
    cand = free[np.abs(d - target_dist_px) < 15.0]
    pick = cand[len(cand) // 2] if len(cand) else free[int(np.argmax(d))]
    return int(pick[0]), int(pick[1])


def xy(r, c):
    # centre the whole scene on the world origin so it sits on the ground plane
    # and aligns with Gazebo's grid. rf_seeker uses the same mapping.
    return (c - 127.5) * RES, (127.5 - r) * RES


def box_model(name, cx, cy, sx, sy, sz):
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{cx:.2f} {cy:.2f} {sz/2:.2f} 0 0 0</pose>
      <link name="l">
        <collision name="c"><geometry><box><size>{sx:.2f} {sy:.2f} {sz:.2f}</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{sx:.2f} {sy:.2f} {sz:.2f}</size></box></geometry>
          <material><ambient>0.5 0.5 0.55 1</ambient><diffuse>0.6 0.6 0.65 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def rover_model(cx, cy):
    # Diff-drive base. Wheels radius 0.3 with centres at z=0.3 so their bottoms
    # rest exactly on the ground (z=0); the caster sphere is the same height.
    # Earlier the wheels spawned ~0.15 m underground, which made the body jitter.
    return f"""    <model name="{ROVER}">
      <pose>{cx:.2f} {cy:.2f} 0 0 0 0</pose>
      <link name="chassis">
        <pose>0 0 0.40 0 0 0</pose>
        <inertial><mass>10.0</mass>
          <inertia><ixx>0.9</ixx><iyy>2.2</iyy><izz>3.0</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="c"><geometry><box><size>1.6 1.0 0.3</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>1.6 1.0 0.3</size></box></geometry>
          <material><ambient>0.1 0.4 0.8 1</ambient><diffuse>0.1 0.5 0.9 1</diffuse></material>
        </visual>
        <visual name="beacon"><pose>0 0 8 0 0 0</pose>
          <geometry><cylinder><radius>0.7</radius><length>16</length></cylinder></geometry>
          <material><ambient>0 0.8 1 1</ambient><diffuse>0 0.9 1 1</diffuse><emissive>0 0.7 1 1</emissive></material>
        </visual>
        <visual name="beacontop"><pose>0 0 16.5 0 0 0</pose>
          <geometry><sphere><radius>1.6</radius></sphere></geometry>
          <material><ambient>0 1 1 1</ambient><diffuse>0 1 1 1</diffuse><emissive>0 0.9 1 1</emissive></material>
        </visual>
      </link>
      <link name="lw">
        <pose>0 0.6 0.30 -1.5708 0 0</pose>
        <inertial><mass>1.0</mass><inertia><ixx>0.03</ixx><iyy>0.03</iyy><izz>0.045</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><surface><friction><ode><mu>1.5</mu><mu2>1.5</mu2></ode></friction></surface>
          <geometry><cylinder><radius>0.30</radius><length>0.15</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.30</radius><length>0.15</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient></material></visual>
      </link>
      <link name="rw">
        <pose>0 -0.6 0.30 -1.5708 0 0</pose>
        <inertial><mass>1.0</mass><inertia><ixx>0.03</ixx><iyy>0.03</iyy><izz>0.045</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><surface><friction><ode><mu>1.5</mu><mu2>1.5</mu2></ode></friction></surface>
          <geometry><cylinder><radius>0.30</radius><length>0.15</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.30</radius><length>0.15</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient></material></visual>
      </link>
      <link name="caster">
        <pose>-0.65 0 0.30 0 0 0</pose>
        <inertial><mass>0.3</mass><inertia><ixx>0.005</ixx><iyy>0.005</iyy><izz>0.005</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface>
          <geometry><sphere><radius>0.30</radius></sphere></geometry></collision>
        <visual name="v"><geometry><sphere><radius>0.30</radius></sphere></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient></material></visual>
      </link>
      <joint name="lwj" type="revolute"><parent>chassis</parent><child>lw</child>
        <axis><xyz>0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>
      <joint name="rwj" type="revolute"><parent>chassis</parent><child>rw</child>
        <axis><xyz>0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>
      <joint name="cj" type="ball"><parent>chassis</parent><child>caster</child></joint>

      <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
        <left_joint>lwj</left_joint>
        <right_joint>rwj</right_joint>
        <wheel_separation>1.2</wheel_separation>
        <wheel_radius>0.30</wheel_radius>
        <topic>/model/{ROVER}/cmd_vel</topic>
        <odom_topic>/model/{ROVER}/odometry</odom_topic>
        <tf_topic>/model/{ROVER}/tf</tf_topic>
        <frame_id>odom</frame_id>
        <child_frame_id>{ROVER}</child_frame_id>
        <odom_publish_frequency>30</odom_publish_frequency>
        <max_linear_acceleration>3.0</max_linear_acceleration>
      </plugin>
    </model>"""


def build_gui():
    """A clean GUI layout: camera framed top-down-ish over the whole 435x435 m
    scene (so the rover, buildings and emitter are all in view on launch),
    plus the standard view/markers/controls plugins. Served to `gz sim -g`."""
    # world centre ~ (216,216); camera sits south of it, up high, angled north.
    cam = "0 -210 320 0 0.95 1.5708"
    return f"""    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui>
          <title>RF Seeker — live</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="string" key="state">docked</property>
        </gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.6 0.6 0.6</ambient_light>
        <background_color>0.7 0.8 0.9</background_color>
        <camera_pose>{cam}</camera_pose>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager"/>
      <plugin filename="InteractiveViewControl" name="Interactive view control"/>
      <plugin filename="CameraTracking" name="Camera Tracking"/>
      <plugin filename="MarkerManager" name="Marker manager"/>
      <plugin filename="SelectEntities" name="Select entities"/>
      <plugin filename="VisualizationCapabilities" name="Visualization capabilities"/>
      <plugin filename="WorldControl" name="World control">
        <gz-gui>
          <title>World control</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">72</property>
          <property type="double" key="width">121</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View"><line own="left" target="left"/><line own="bottom" target="bottom"/></anchors>
        </gz-gui>
        <play_pause>true</play_pause>
        <step>true</step>
        <start_paused>false</start_paused>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <gz-gui>
          <title>World stats</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">110</property>
          <property type="double" key="width">290</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View"><line own="right" target="right"/><line own="bottom" target="bottom"/></anchors>
        </gz-gui>
        <sim_time>true</sim_time><real_time>true</real_time><real_time_factor>true</real_time_factor>
      </plugin>
    </gui>"""


def build_world(mask, tx_pix, start_pix, decim=1, no_buildings=False):
    if no_buildings:
        rects = []
    elif decim > 1:
        Hp = N // decim
        pooled = mask[:Hp*decim, :Hp*decim].reshape(Hp, decim, Hp, decim).max(axis=(1, 3))
        rects = [(r*decim, c*decim, h*decim, w*decim)
                 for (r, c, h, w) in merge_rectangles(pooled)]
    else:
        rects = merge_rectangles(mask)
    blocks = []
    for i, (r, c, h, w) in enumerate(rects):
        cx, cy = xy(r + h / 2.0 - 0.5, c + w / 2.0 - 0.5)
        blocks.append(box_model(f"b{i}", cx, cy, w * RES, h * RES, WALL_H))
    ex, ey = xy(*tx_pix)
    sx, sy = xy(*start_pix)
    emitter = f"""    <model name="emitter">
      <static>true</static>
      <pose>{ex:.2f} {ey:.2f} 0 0 0 0</pose>
      <link name="l">
        <visual name="pillar"><pose>0 0 25 0 0 0</pose>
          <geometry><cylinder><radius>3</radius><length>50</length></cylinder></geometry>
          <material><ambient>1 0.35 0 1</ambient><diffuse>1 0.45 0 1</diffuse><emissive>1 0.4 0 1</emissive></material>
        </visual>
        <visual name="ball"><pose>0 0 52 0 0 0</pose>
          <geometry><sphere><radius>5</radius></sphere></geometry>
          <material><ambient>1 0.85 0 1</ambient><diffuse>1 0.85 0 1</diffuse><emissive>1 0.75 0 1</emissive></material>
        </visual>
      </link>
    </model>"""

    return f"""<?xml version="1.0"?>
<sdf version="1.10">
  <world name="rfsim">
    <physics name="1ms" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.7 0.8 0.9 1</background></scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows><direction>-0.4 0.3 -0.9</direction>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular>
    </light>
    <model name="ground"><static>true</static><link name="l">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>700 700</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>700 700</size></plane></geometry>
        <material><ambient>0.25 0.27 0.25 1</ambient><diffuse>0.3 0.32 0.3 1</diffuse></material></visual>
    </link></model>
{emitter}
{chr(10).join(blocks)}
{rover_model(sx, sy)}
{build_gui()}
  </world>
</sdf>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="rfsim.world")
    ap.add_argument("--start-dist", type=float, default=150.0)
    ap.add_argument("--decim", type=int, default=4,
                    help="downsample obstacles by this factor for fast rendering (1 = full res)")
    ap.add_argument("--no-buildings", action="store_true",
                    help="omit physical obstacles entirely (planner still avoids them)")
    a = ap.parse_args()
    mask, tx_pix = load_mask(a.data)
    start_pix = free_start(mask, tx_pix, a.start_dist)
    world = build_world(mask, tx_pix, start_pix, decim=a.decim, no_buildings=a.no_buildings)
    open(a.out, "w").write(world)
    import json, os
    scene = {"data": os.path.abspath(a.data), "res": RES,
             "start_pixel": list(start_pix), "start_xy": list(xy(*start_pix)),
             "tx_pixel": list(tx_pix), "tx_xy": list(xy(*tx_pix))}
    json.dump(scene, open("scene.json", "w"), indent=2)
    print(f"Wrote {a.out} and scene.json")
    print(f"  emitter pixel {tx_pix} -> world {tuple(round(v,1) for v in xy(*tx_pix))}")
    print(f"  rover  start {start_pix} -> world {tuple(round(v,1) for v in xy(*start_pix))}")
    nblk = world.count(chr(60) + 'model name="b')
    print(f"  obstacle blocks: {nblk}")


if __name__ == "__main__":
    main()