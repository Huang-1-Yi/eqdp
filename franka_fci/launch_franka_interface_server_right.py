import zerorpc
from polymetis import RobotInterface
import scipy.spatial.transform as st
import numpy as np
import torch
import time
from polymetis import GripperInterface

# pip install pynput==1.7.6
from pynput.keyboard import Key, KeyCode, Listener
from collections import defaultdict
from threading import Lock



class FrankaInterface:
    def __init__(self):
        # 连接右侧机械臂
        self.robot_right = RobotInterface(
            ip_address="192.168.3.12",
            port=50054
        )
        self.gripper_open = False

        # 初始位置
        self.move_to_start()

        self.gripper_flag = True
        if self.gripper_flag == True:
            self.gripper = GripperInterface(
                ip_address="192.168.3.12",
                port=50053
            )

    # cos pi/8, sin pi/8
    def move_to_start(self, ee_pose = [0.35, -0.0, 0.70], ee_orientation = [0.92388, -0.38268, -0.0, -0.0 ]):
        # 初始位置
        joint_positions_desired3 = [0.002043521963059902, -0.7858179807662964, 0.0025901622138917446, -2.356106758117676, 0.002696990268304944, 1.5775476694107056, 0.7776400446891785]
        self.move_to_joint_positions(joint_positions_desired3, time_to_go=5.0)
        # time.sleep(0.1)
        print("",self.get_ee_pose())
        print("",self.get_joint_positions())
        
        # self.ee_pose0 = [0.35, -0.0, 0.70]
        self.robot_right.move_to_ee_pose(position=ee_pose, orientation=ee_orientation)
        
        time.sleep(0.5)
        data1= self.robot_right.get_ee_pose()
        pos1 = data1[0].numpy()
        quat_xyzw1 = data1[1].numpy()
        print("pos:",pos1,"quat:",quat_xyzw1)
        print("get_joint_positions：",self.get_joint_positions())
        print("get_ee_pose：",self.get_ee_pose())
        return True

    def get_ee_pose(self):
        data1= self.robot_right.get_ee_pose()
        pos1 = data1[0].numpy()
        quat_xyzw1 = data1[1].numpy()
        rot_vec1 = st.Rotation.from_quat(quat_xyzw1).as_rotvec()
        data_return = np.concatenate([pos1, rot_vec1]).tolist()
        return data_return

    def get_joint_positions(self):
        return self.robot_right.get_joint_positions().numpy().tolist()

    def get_joint_velocities(self):
        return self.robot_right.get_joint_velocities().numpy().tolist()

    def move_to_joint_positions(self, positions, time_to_go):

        self.robot_right.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go
        )

    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot_right.start_cartesian_impedance(
            Kx=torch.Tensor(Kx),
            Kxd=torch.Tensor(Kxd)
        )
        print("robot_right!!!start_cartesian_impedance")

    def update_desired_ee_pose(self, pose):
        arm_pose = np.asarray(pose)
        self.robot_right.update_desired_ee_pose(
            position=torch.Tensor(arm_pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(arm_pose[3:6]).as_quat())
        )

    def terminate_current_policy(self):
        # self.robot_right.terminate_current_policy()
        print("terminate_current_policy")
        # exit()


    def get_gripper_width(self):
        """
        self.gripper.get_state():
            width timestamp {
            seconds: 1747226056
            nanos: 696573703
            }
            width: 0.07999776303768158
            prev_command_successful: true
        """
        raw_width = self.gripper.get_state().width
        width = round(raw_width, 4)

        print("width",width)
        return np.array([width]).tolist()

    def open_gripper(self, width, speed=0.02, block=True):#, epsilon_inner_value=-1.0,epsilon_outer_value=-1.0
        print("open_gripper, width:",width)
        if self.gripper_open != True:
            if self.gripper_flag == False:
                print("open_gripper, width:",width,"speed:",speed)
                self.gripper.goto(width=width,speed =speed,force=0.01,blocking=block)
                self.gripper_flag = True
            # else:
            #     print("error! the gripper is still True")
            #     print("flag==",self.gripper_flag,"open_gripper==",width)
   
    def close_gripper(self, width, speed=0.03, block=True, epsilon_inner_value=-1.0,epsilon_outer_value=-1.0):
        print("close_gripper, width:",width)
        if self.gripper_open != True:
            if self.gripper_flag == True:
                print("close_gripper, width:",width,"speed:",speed)
                width = 0 
                self.gripper.grasp(grasp_width=width,speed =speed,force=0.01,blocking=block,epsilon_inner=epsilon_inner_value,epsilon_outer=epsilon_outer_value)
                self.gripper_flag = False
            # else:
            #     print("flag==",self.gripper_flag,"close_gripper==",width)
        #return state

    def keyboard(self):
        print("keyboard")


s = zerorpc.Server(FrankaInterface())
s.bind("tcp://0.0.0.0:4242")
s.run()
