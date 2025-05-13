"""
先用 RobomimicObsConverter 生成观测，
再用 RobomimicAbsoluteActionConverter 处理动作


RobomimicAbsoluteActionConverter
动作空间转换​​：将相对动作（delta动作）转换为绝对坐标系下的目标位置/姿态。
- 创建两个环境实例：env（支持delta动作）和 abs_env（绝对动作）
- 关闭 control_delta 以禁用相对控制
关键方法
convert_actions()
- 解析delta动作序列
- 通过逆运动学计算绝对目标位姿
- 保留夹爪开合状态
evaluate_rollout_error()
- 对比转换后动作的执行轨迹与原数据的一致性
- 计算位置/姿态误差
典型使用场景
- 将基于相对动作的策略（如PD控制器）迁移到绝对坐标系
- 验证动作模型在不同控制器下的兼容性
解决的是动作表示形式的兼容性问题​​，适用于需要精确控制绝对位姿的场景

RobomimicObsConverter
​​观察空间扩展​​：从状态数据生成包含多模态观察（如图像、深度等）的观测数据。
- 创建单一环境实例，配置多摄像头（如 birdview, agentview）
- 指定图像分辨率和传感器类型（RGB/深度）
关键方法 convert_obs()
- 从初始状态生成多模态观测
- 包含RGB图像、深度图等传感器数据
无显式验证，直接删除不需要的观测（如 del obss['birdview_depth']）
典型使用场景
- 为视觉策略（如CNN+强化学习）生成训练所需的观测数据
- 数据增强或传感器模拟
解决的是观测信息丰富性问题​​，适用于依赖视觉或其他传感器输入的算法。

绝对动作转换​
converter = RobomimicAbsoluteActionConverter("dataset.hdf5")
abs_actions = converter.convert_idx(0)  # 转换第0条演示数据

观察数据生成​
converter = RobomimicObsConverter("dataset.hdf5")
obs_dict = converter.convert_idx(0)  # 生成第0条演示的观测
# obs_dict包含 agentview_image, robot0_eye_in_hand_image 等键
"""

import numpy as np
import copy

import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.tensor_utils as TensorUtils  # eqdp导入
from scipy.spatial.transform import Rotation

from robomimic.config import config_factory

# ​​动作空间转换​​：将相对动作（delta动作）转换为绝对坐标系下的目标位置/姿态。
class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert len(env.env.robots) in (1, 2)

        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert not abs_env.env.robots[0].controller.use_delta

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, 'r')
    
    def __len__(self):
        return len(self.file['data'])

    def convert_actions(self, 
            states: np.ndarray, 
            actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1],-1,7)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]
        for i in range(len(states)):
            _ = env.reset_to({'states': states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
            
                # read pos and ori from robots
                controller = robot.controller
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(
                    controller.goal_ori).as_rotvec()

        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]

        delta_error_info = self.evaluate_rollout_error(
            env, states, actions, robot0_eef_pos, robot0_eef_quat, 
            metric_skip_steps=eval_skip_steps)
        abs_error_info = self.evaluate_rollout_error(
            abs_env, states, abs_actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps)

        info = {
            'delta_max_error': delta_error_info,
            'abs_max_error': abs_error_info
        }
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(env, 
            states, actions, 
            robot0_eef_pos, 
            robot0_eef_quat, 
            metric_skip_steps=1):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({'states': states[0]})
        for i in range(len(states)):
            obs = env.reset_to({'states': states[i]})
            obs, reward, done, info = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_states.append(env.get_state()['states'])
            rollout_next_eef_pos.append(obs['robot0_eef_pos'])
            rollout_next_eef_quat.append(obs['robot0_eef_quat'])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            'state': max_next_state_diff,
            'pos': max_next_eef_pos_dist,
            'rot': max_next_eef_rot_dist
        }
        return info

# eqdo创建的RobomimicObsConverter
# ​​观察空间扩展​​：从状态数据生成包含多模态观察（如图像、深度等）的观测数据。
class RobomimicObsConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        # default BC config
        # config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        # ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        # env_meta['env_kwargs']['camera_names'] = ['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']

        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            # camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'agentview_full', 'robot0_robotview', 'robot0_eye_in_hand'], 
            camera_names=['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand'], 
            camera_height=84, 
            camera_width=84, 
            reward_shaping=False,
        )
        # env = EnvUtils.create_env_from_metadata(
        #     env_meta=env_meta,
        #     render=True, 
        #     render_offscreen=True,
        #     use_image_obs=True,
        # )

        self.env = env
        self.file = h5py.File(dataset_path, 'r')
    
    def __len__(self):
        return len(self.file['data'])

    def convert_obs(self, initial_state, states):
        obss = []
        self.env.reset()
        obs = self.env.reset_to(initial_state)
        obss.append(obs)
        for i in range(1, len(states)):
            obs = self.env.reset_to({'states': states[i]})
            obss.append(obs)
        return TensorUtils.list_of_flat_dict_to_dict_of_list(obss)

    def convert_idx(self, idx):
        file = self.file
        demo = file[f'data/demo_{idx}']
        # input

        states = demo['states'][:]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo.attrs["model_file"]

        # generate abs actions
        obss = self.convert_obs(initial_state, states)
        del obss['birdview_image']
        del obss['birdview_depth']
        del obss['agentview_depth']
        del obss['sideview_image']
        del obss['sideview_depth']
        del obss['robot0_eye_in_hand_depth']
        return obss
