"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from real_world.real_env_franka import RealEnvFranka as RealEnv
from real_world.spacemouse_shared_memory import Spacemouse
from eqdp.common.precise_sleep import precise_wait
from real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from eqdp.common.pytorch_util import dict_apply
from eqdp.workspace.base_workspace import BaseWorkspace
from eqdp.policy.base_image_policy import BaseImagePolicy
from eqdp.common.cv2_util import get_image_transform
from real_world.keystroke_counter          import ( KeystrokeCounter, Key, KeyCode )


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency):
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    elif 'robomimic' in cfg.name:
        # BCRNN model
        policy: BaseImagePolicy
        policy = workspace.model

        device = torch.device('cuda')
        policy.eval().to(device)

        # BCRNN always has action horizon of 1
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)

    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096

        device = torch.device('cuda')
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    return_begin =False
    init_positon = np.array([0.3551,-0.0026,0.6935,-2.8691,1.2213,0.0139])
    epsilon = 0.02                                      # 位置误差容忍阈值 (单位：米)0.004
    step_size = 0.02                                    # 调整步长0.01
    epsilon_rot = 0.0175                                # 角度死区阈值 (约1度，单位：弧度)
    max_step_angle = 0.07                               # 最大单步调整角度 (约5度，单位：弧度)


    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager) as sm, \
            RealEnv(
                    output_dir=output, 
                    robot_ip=robot_ip, 
                    frequency=frequency,
                    n_obs_steps=n_obs_steps,
                    obs_image_resolution=obs_res,
                    obs_float32=True,
                    init_joints=init_joints,
                    enable_multi_cam_vis=True,
                    record_raw_video=True,
                    # number of threads per camera view for video recording (H.264)
                    thread_per_video=3,
                    # video recording quality, lower is better (but slower).
                    video_crf=21,
                    shm_manager=shm_manager,
                    enable_sam2 = False,
            ) as env:
            cv2.setNumThreads(1)

            # # Should be the same as demo
            # # realsense exposure
            # env.realsense.set_exposure(exposure=120, gain=0)
            # # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            env_enable_sam2 = env.enable_sam2
            
            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            # print("Observation keys:", obs.keys())  # 确认包含stage
            # print("Stage shape:", obs['stage'].shape)  # 应为(n_obs_steps, 1)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                del result

            print('Ready!')
            stage = 0
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                # target_pose = state['TargetTCPPose']
                target_pose = state['ActualTCPPose']
                target_gripper = state['ActualGripperstate']

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    obs = env.get_obs()
                    # print("Observation keys:", obs.keys())  # 确认包含stage
                    # print("Stage shape:", obs['stage'].shape)  # 应为(n_obs_steps, 1)                    # visualize
                    
                    if env_enable_sam2:
                        mask = obs[f'camera_{vis_camera_idx}'][-1,:,:].copy()
                        vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)    # 转换为3通道BGR图像
                        cv2.imshow('Mask Image', vis_img[:, :, ::-1])       # 转为RGB格式（仅当显示库需要时）显示RGB图像
                    else:
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1].copy()
                        cv2.imshow('Rgb Image', vis_img[...,::-1])          # 修复颜色

                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('q'):                              # 退出程序
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord('w'):                            # 结束人类控制循环，切换策略控制循环
                        stage = 0
                        return_begin = False
                        print("C pressed! Exiting human control loop.")
                        cv2.waitKey(1)                                      # 强制刷新事件队列
                        break
                    elif key_stroke == 32:                                  # Space key# 新增空格键处理
                        print("space pressed!")
                        stage +=1
                    elif key_stroke == ord('r'):
                        return_begin = True
                        print("r被按下, Return to initial pose triggered")

                    gripper_state = (stage) % 2 *0.08
                    # print("Stage: ", stage, "Gripper state: ", gripper_state)
                    gripper_now = obs['robot_gripper'][-1]                  # 获取当前夹爪状态
                    print("gripper_state",gripper_state,"gripper_now",gripper_now)
                    
                    precise_wait(t_sample)
                    
                    if return_begin:
                        # ===== 位置控制 =====
                        for i in range(3):
                            # 计算当前位置与目标的差值
                            error = init_positon[i] - target_pose[i]
                            abs_error = abs(error)
                            if abs_error > epsilon:                         # 差值超过阈值时，按比例调整步长
                                adjustment = np.sign(error) * min(step_size, abs_error)
                                target_pose[i] += adjustment
                            else:
                                # 差值小于阈值时，直接设为目标值
                                target_pose[i] = init_positon[i]
                        # ===== 旋转控制 =====
                        current_rot = st.Rotation.from_rotvec(target_pose[3:6])
                        target_rot = st.Rotation.from_rotvec(init_positon[3:6])
                        rel_rot = target_rot * current_rot.inv()
                        rel_angle = rel_rot.magnitude()
                        if rel_angle > epsilon_rot:
                            step_angle = min(rel_angle, max_step_angle)
                            adjust_rot = st.Rotation.from_rotvec(
                                rel_rot.as_rotvec() * (step_angle / rel_angle)
                            )
                            new_rot = adjust_rot * current_rot
                            target_pose[3:6] = new_rot.as_rotvec()
                        else:
                            target_pose[3:6] = init_positon[3:6]
                    else:   # 空间鼠标控制
                        sm_state = sm.get_motion_state_transformed()
                        # print(sm_state)
                        dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                        drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
                        drot = st.Rotation.from_euler('xyz', drot_xyz)
                        target_pose[:3] += dpos
                        target_pose[3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[3:])).as_rotvec()

                    # 位置拼接
                    actions =np.zeros(7)
                    actions[:6] = target_pose
                    actions[6] = gripper_state
                    # print("Target pose: ", target_pose)
                    # print("Gripper state: ", gripper_state)

                    #  命令执行
                    env.exec_actions(
                        actions=[actions], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        stages=[[stage]],
                        )
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    actions =np.zeros(7)
                    actions[:6] = target_pose
                    actions[6] = target_gripper

                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        
                        # stage = obs['stage'][-1]
                        gripper_state = stage = obs['robot_gripper'][-1]
                        print("Stage: ", stage, "Gripper state: ", gripper_state)
                    
                        # print("Observation keys:", obs.keys())  # 确认包含stage
                        # print("Stage shape:", obs['stage'].shape)  # 应为(n_obs_steps, 1)


                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, 
                                shape_meta=cfg.task.shape_meta
                                )
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        # if delta_action:
                        #     assert len(action) == 1
                        #     if perv_target_pose is None:
                        #         perv_target_pose = obs['robot_eef_pose'][-1]
                        #     this_target_pose = perv_target_pose.copy()
                        #     this_target_pose[[0,1]] += action[-1]
                        #     perv_target_pose = this_target_pose
                        #     this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        # else:                                           # 7替换掉len(target_pose)
                        #     this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                        #     this_target_poses[:] = target_pose
                        #     this_target_poses[:,[0,1]] = action

                        if delta_action:
                            assert len(action) == 1
                            if perv_target_pose is None:
                                perv_target_pose = obs['robot_eef_pose'][-1]
                            this_target_pose = perv_target_pose.copy()
                            this_target_pose += action[-1]
                            perv_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        else:                                           # 7替换掉len(target_pose)
                            this_target_poses = np.zeros((len(action), len(actions)), dtype=np.float64)
                            this_target_poses[:] = actions 
                            this_target_poses[:,:] = action

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # clip actions
                        # this_target_poses[:,:2] = np.clip(
                        #     this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])

                        # execute actions
                        print(f"Executing {len(this_target_poses)} steps of actions. Gripper state== {gripper_state}")
                        this_target_poses[:,6] = gripper_state
                        print("gripper_wid==",this_target_poses[:,6])
                        print("action[0,:]",this_target_poses[0,:])
                        print("action[6,:]",this_target_poses[0,:])
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # # visualize
                        # episode_id = env.replay_buffer.n_episodes
                        # vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                        # text = 'Episode: {}, Time: {:.1f}'.format(
                        #     episode_id, time.monotonic() - t_start
                        # )
                        # cv2.putText(
                        #     vis_img,
                        #     text,
                        #     (10,20),
                        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #     fontScale=0.5,
                        #     thickness=1,
                        #     color=(255,255,255)
                        # )
                        # cv2.imshow('default', vis_img[...,::-1])

                        if env_enable_sam2:
                            mask = obs[f'camera_{vis_camera_idx}'][-1,:,:].copy()
                            vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)    # 转换为3通道BGR图像
                            cv2.imshow('Mask Image', vis_img[:, :, ::-1])       # 转为RGB格式（仅当显示库需要时）显示RGB图像
                        else:
                            vis_img = obs[f'camera_{vis_camera_idx}'][-1].copy()
                            cv2.imshow('Rgb Image', vis_img[...,::-1])            # 修复颜色

                        # press_events = key_counter.get_press_events()   # 获取按键事件
                        # for key_stroke in press_events:                 # 遍历按键事件
                        #     if key_stroke == KeyCode(char='s'):
                        #         env.end_episode()                       # 结束当前集
                        #         key_counter.clear()                     # 清除按键计数器
                        #         is_recording = False                    # 设置录制标志为False
                        #         return_begin = False
                        #         return_flag = True
                        #         print('Stopped.')                       # 打印停止消息

                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            stage = 0
                            env.end_episode()
                            print('Stopped.')
                            break

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                        curr_pose = obs['robot_eef_pose'][-1]
                        dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                        if dist < 0.03:
                            # in termination area
                            curr_timestamp = obs['timestamp'][-1]
                            if term_area_start_timestamp > curr_timestamp:
                                term_area_start_timestamp = curr_timestamp
                            else:
                                term_area_time = curr_timestamp - term_area_start_timestamp
                                if term_area_time > 0.5:
                                    terminate = True
                                    print('Terminated by the policy!')
                        else:
                            # out of the area
                            term_area_start_timestamp = float('inf')

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
