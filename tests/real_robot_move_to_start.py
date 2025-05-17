import numpy as np
import zerorpc



class FrankaInterface:
    def __init__(self, ip='192.168.0.168', port=4242):
        self.server_robot = zerorpc.Client(heartbeat=20)
        self.server_robot.connect(f"tcp://{ip}:{port}")

    def open_gripper(self, width: float, speed: float = 0.02):
        if width < 0 or speed <= 0:
            raise ValueError("宽度和速度必须为正值。")
        # print(f"Opening gripper: width={width}, speed={speed}")
        self.server_robot.open_gripper(width, speed)

    def close_gripper(self, width: float = 0.0, force: float = 20.0, speed: float = 0.02):
        if force <= 0 or speed <= 0:
            raise ValueError("力和速度必须为正值。")
        # print(f"Closing gripper: width={width} force={force}, speed={speed}")
        self.server_robot.close_gripper(width, speed, force)


    def get_ee_pose(self):
        ee_pose = np.array(self.server_robot.get_ee_pose())
        return ee_pose

    def get_joint_positions(self):
        return np.array(self.server_robot.get_joint_positions())

    def get_joint_velocities(self):
        return np.array(self.server_robot.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server_robot.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server_robot.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )

    # to do
    def get_gripper_width(self):
        # gripper_flag = np.array([self.server.get_gripper_width()], dtype=np.float32)
        gripper_flag = np.array(self.server_robot.get_gripper_width())
        return gripper_flag
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server_robot.update_desired_ee_pose(pose.tolist())

    def terminate_current_policy(self):
        self.server_robot.terminate_current_policy()

    def close(self):
        self.server_robot.close()

    def move_to_start(self, ee_pose: np.ndarray, ee_orientation: np.ndarray):# [0.35, 0.0, 0.7], [0.92388, -0.38268, 0.0, 0.0]
        self.server_robot.move_to_start(ee_pose.tolist())


if __name__ == "__main__":
    # 创建一个FrankaInterface实例
    robot = FrankaInterface("192.168.0.168", 4242)
    # som用
    # data = robot.move_to_start(np.array([0.5, 0.0, 0.75]),np.array([0.92388, -0.38268, 0.0, 0.0])) 
    data = robot.move_to_start(np.array([0.5, 0.0, 0.75]),np.array([0.92388, -0.38268, 0.0, 0.0]))   
    print("移动到起始位置:", data)
    # curr_pose = robot.get_ee_pose()      #获取当前末端执行器姿态
    # print("当前末端执行器姿态:", curr_pose)
