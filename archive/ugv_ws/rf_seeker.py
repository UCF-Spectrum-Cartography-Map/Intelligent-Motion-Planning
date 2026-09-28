"""
rf_seeker.py — ROS 2 node driving the Gazebo rover with the GNN->PPO planner.

Loop: read rover pose -> pixel -> SeekerBrain.plan() -> next waypoint ->
drive cmd_vel toward it -> repeat, until within reach of the emitter.

Reads scene.json (written by make_world.py) for the start pose and scene path.
Run inside the container after make_world.py, alongside the Gazebo world.
"""
import json
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
# seeker_brain (which imports torch) is loaded lazily INSIDE __init__, AFTER the
# ROS node exists. Importing torch before rcl node creation can deadlock the
# DDS participant against torch's thread pool.

RES = 1.7
HALF = 127.5          # grid centre; world is centred on it (matches make_world.xy)
ROVER_TOPIC = "/model/rover"


def pixel_of_xy(x, y):
    return (int(round(HALF - y / RES)), int(round(x / RES + HALF)))


def xy_of_pixel(r, c):
    return ((c - HALF) * RES, (HALF - r) * RES)


class RFSeeker(Node):
    def __init__(self):
        super().__init__("rf_seeker")
        print("[rf_seeker] node created; now importing torch + loading models "
              "(~30-60s)...", flush=True)
        from seeker_brain import SeekerBrain
        self._SeekerBrain = SeekerBrain
        print("[rf_seeker] torch import done.", flush=True)
        self.declare_parameter("scene", "scene.json")
        self.declare_parameter("gnn", "GNN_best.pth")
        self.declare_parameter("ppo", "ppo_GNN.pth")
        sc = json.load(open(self.get_parameter("scene").value))
        self.start_xy = tuple(sc["start_xy"])
        self.tx_xy = tuple(sc["tx_xy"])
        print(f"[rf_seeker] scene loaded ({sc['data']}); building SeekerBrain "
              f"(loads 2 models, ~20-40s)...", flush=True)
        self.brain = self._SeekerBrain(sc["data"],
                                       self.get_parameter("gnn").value,
                                       self.get_parameter("ppo").value)
        print("[rf_seeker] models loaded; resetting episode...", flush=True)
        self.brain.reset(tuple(sc["start_pixel"]))
        print("[rf_seeker] episode ready; subscribing to odometry.", flush=True)

        self.pose = None
        self.target_w = None
        self.done = False
        self.path = []

        self.cmd = self.create_publisher(Twist, f"{ROVER_TOPIC}/cmd_vel", 10)
        self.mk = self.create_publisher(Marker, "/rf_path", 10)
        self.recon_pub = self.create_publisher(Image, "/rf_recon", 1)
        self.markers_pub = self.create_publisher(MarkerArray, "/rf_markers", 1)
        self.tf = TransformBroadcaster(self)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, "/rf_map", latched)
        self.cloud_pub = self.create_publisher(PointCloud2, "/rf_recon_cloud", 1)
        self.publish_map()
        self.create_subscription(Odometry, f"{ROVER_TOPIC}/odometry", self.on_odom, 10)
        self.create_timer(0.1, self.control)
        print("[rf_seeker] up; waiting for odometry ...", flush=True)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # DiffDrive odometry is relative to the spawn pose (spawn yaw = 0),
        # so add the absolute spawn offset to get world coordinates.
        x = self.start_xy[0] + p.x
        y = self.start_xy[1] + p.y
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self.pose = (x, y, yaw)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "rover"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf.sendTransform(t)

    def control(self):
        if self.pose is None or self.done:
            return
        x, y, yaw = self.pose
        pix = pixel_of_xy(x, y)

        # Replan when we have no waypoint or have reached the current one.
        if self.target_w is None or math.hypot(self.target_w[0] - x, self.target_w[1] - y) < 1.5:
            target, done, info = self.brain.plan(pix)
            self.target_w = xy_of_pixel(*target)
            self.done = done
            self.path.append((x, y))
            self.publish_path()
            est_world = None
            if self.brain.last_recon is not None:
                gy, gx = np.unravel_index(int(np.argmin(self.brain.last_recon)),
                                          self.brain.last_recon.shape)
                est_world = xy_of_pixel(gy, gx)
            self.publish_markers(x, y, yaw, est_world)
            self.publish_recon(pix)
            self.publish_recon_cloud()
            print(f"[rf_seeker] pix={pix} dist_to_emitter={info['dist_px']:.0f}px", flush=True)
            if done:
                self.cmd.publish(Twist())
                print("[rf_seeker] REACHED emitter — stopping.", flush=True)
                return

        # Drive toward the waypoint: turn AND move at once; forward speed scaled
        # by how well we're aimed (cos of heading error), so it never dithers.
        tx, ty = self.target_w
        d = math.hypot(tx - x, ty - y)
        heading = math.atan2(ty - y, tx - x)
        err = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        cmd = Twist()
        cmd.angular.z = max(-2.0, min(2.0, 2.5 * err))
        cmd.linear.x = max(0.0, min(1.5, 1.3 * d)) * max(0.0, math.cos(err))
        self.cmd.publish(cmd)

    def publish_recon(self, rover_rc):
        recon = self.brain.last_recon
        if recon is None:
            return
        v = np.clip(recon, 0.0, 1.0).astype(np.float32)
        g = ((1.0 - v) * 255).astype(np.uint8)        # bright where signal is strong (near emitter)
        img = np.stack([g, g, g], axis=-1).copy()

        def mark(rc, color, rad=3):
            r, c = int(rc[0]), int(rc[1])
            img[max(0, r - rad):r + rad + 1, max(0, c - rad):c + rad + 1] = color

        gy, gx = np.unravel_index(int(np.argmin(v)), v.shape)
        mark(self.brain.tx_pixel, [0, 220, 0])        # true emitter  (green)
        mark((gy, gx), [230, 0, 0])                   # GNN estimate  (red)
        mark(rover_rc, [0, 200, 255])                 # rover         (cyan)

        m = Image()
        m.header.frame_id = "map"
        m.height, m.width = 256, 256
        m.encoding = "rgb8"
        m.is_bigendian = 0
        m.step = 256 * 3
        m.data = img.tobytes()
        self.recon_pub.publish(m)

    def publish_path(self):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "rf_path"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 2.5
        m.color.g = 0.8
        m.color.b = 1.0
        m.color.a = 1.0
        for (px, py) in self.path:
            pt = Point()
            pt.x, pt.y, pt.z = float(px), float(py), 0.5
            m.points.append(pt)
        self.mk.publish(m)

    def publish_recon_cloud(self):
        """Reconstruction as a flat colored ground overlay. Points are placed
        ONLY over free cells so the dark building footprint shows through; the
        ramp is a muted slate->amber kept dim so rover/emitter/path/markers
        (all drawn above it) stay clearly visible."""
        recon = self.brain.last_recon
        if recon is None:
            return
        step = 2
        mask = self.brain.env.building_mask
        rr, cc = np.meshgrid(np.arange(0, 256, step), np.arange(0, 256, step), indexing="ij")
        free = mask[rr, cc] == 0
        rr, cc = rr[free], cc[free]
        x = ((cc - HALF) * RES).astype(np.float32)
        y = ((HALF - rr) * RES).astype(np.float32)
        z = np.full(x.shape, 0.05, dtype=np.float32)
        s = (1.0 - np.clip(recon[rr, cc], 0.0, 1.0)).astype(np.float32)   # signal strength
        R = (35 + s * 125).astype(np.uint32)
        G = (35 + s * 55).astype(np.uint32)
        B = np.full(R.shape, 42, dtype=np.uint32)
        rgb = ((R << 16) | (G << 8) | B).astype(np.uint32)
        N = int(x.size)
        cloud = np.zeros(N, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
        cloud["x"], cloud["y"], cloud["z"] = x, y, z
        cloud["rgb"] = rgb.view(np.float32)
        pc = PointCloud2()
        pc.header.frame_id = "map"
        pc.header.stamp = self.get_clock().now().to_msg()
        pc.height, pc.width = 1, N
        pc.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc.is_bigendian = False
        pc.point_step = 16
        pc.row_step = 16 * N
        pc.is_dense = True
        pc.data = cloud.tobytes()
        self.cloud_pub.publish(pc)

    def publish_map(self):
        mask = self.brain.env.building_mask
        og = OccupancyGrid()
        og.header.frame_id = "map"
        og.header.stamp = self.get_clock().now().to_msg()
        og.info.resolution = RES
        og.info.width = 256
        og.info.height = 256
        # cell (0,0) sits at the SW corner of the recentred world; +x=col, +y=up.
        og.info.origin.position.x = -HALF * RES
        og.info.origin.position.y = -HALF * RES
        og.info.origin.orientation.w = 1.0
        # OccupancyGrid rows go +y (bottom-up); image rows go -y, so flip.
        occ = (np.flipud(mask) > 0).astype(np.int8) * 100
        og.data = occ.flatten().tolist()
        self.map_pub.publish(og)
        print("[rf_seeker] published obstacle OccupancyGrid on /rf_map", flush=True)

    def _sphere(self, mid, wx, wy, wz, rad, rgb):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "rf"; m.id = mid
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = wx, wy, wz
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = rad * 2.0
        m.color.r, m.color.g, m.color.b, m.color.a = rgb[0], rgb[1], rgb[2], 1.0
        return m

    def publish_markers(self, x, y, yaw, est_world):
        arr = MarkerArray()
        arr.markers.append(self._sphere(0, self.tx_xy[0], self.tx_xy[1], 6.0, 7.0, (0.0, 1.0, 0.2)))
        if est_world is not None:
            arr.markers.append(self._sphere(1, est_world[0], est_world[1], 4.0, 5.0, (1.0, 0.1, 0.1)))
        rm = Marker()
        rm.header.frame_id = "map"
        rm.header.stamp = self.get_clock().now().to_msg()
        rm.ns = "rf"; rm.id = 2
        rm.type = Marker.CUBE; rm.action = Marker.ADD
        rm.pose.position.x, rm.pose.position.y, rm.pose.position.z = x, y, 1.5
        rm.pose.orientation.z = math.sin(yaw / 2.0)
        rm.pose.orientation.w = math.cos(yaw / 2.0)
        rm.scale.x, rm.scale.y, rm.scale.z = 8.0, 5.0, 3.0
        rm.color.r, rm.color.g, rm.color.b, rm.color.a = 0.0, 0.8, 1.0, 1.0
        arr.markers.append(rm)
        self.markers_pub.publish(arr)


def main():
    print("[rf_seeker] calling rclpy.init() ...", flush=True)
    rclpy.init()
    print("[rf_seeker] rclpy.init() returned; creating node ...", flush=True)
    node = RFSeeker()
    print("[rf_seeker] node ready; spinning.", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()