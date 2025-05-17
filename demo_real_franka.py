# 版本：2025.05.07
# 作者：黄一

"""
​​旋转角度死区控制​​
    当当前姿态与目标姿态角度差 rel_angle < epsilon_rot（默认1度）时，直接设为目标姿态
    避免微小角度调整造成的震荡
​​轴角增量步长控制​​
    使用 max_step_angle 限制单次最大调整角度（默认5度）
    增量方向与目标方向一致：rel_rot.as_rotvec() * (step_angle / rel_angle)
​​旋转向量归一化​​
    SciPy 的 Rotation 类会自动处理超过 2π 的旋转向量，确保数值稳定性

使用方法：
(robodiff)$ python demo_real_robot.py -o <演示保存目录> --robot_ip <ur5的ip地址>
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

机器人移动：
移动您的SpaceMouse以移动机器人的末端执行器（锁定在xy平面）。
按下SpaceMouse右键以解锁z轴。
按下SpaceMouse左键以启用旋转轴。

录制控制：
点击opencv窗口（确保它是焦点）。
按"W"开始录制。
按"S"停止录制。
按"Q"退出程序。
按"R"回到原位。
按"退格键"删除先前录制的剧集。
"""

# %%
import time
from multiprocessing.managers               import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform              as st
from real_world.real_env_franka            import RealEnvFranka as RealEnv
from real_world.spacemouse_shared_memory    import Spacemouse
from real_world.precise_sleep               import precise_wait
from real_world.keystroke_counter           import ( KeystrokeCounter, Key, KeyCode  )

import open3d as o3d

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="Franka's IP address e.g. 172.16.0.1")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1 / frequency                    # 计算时间步长
    # 使用共享内存管理器、按键计数器、SpaceMouse、真实环境
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
                Spacemouse(shm_manager=shm_manager) as sm, \
                RealEnv(
                    output_dir=output,                          # 实验结果输出目录
                    robot_ip=robot_ip,                          # 机器人IP地址
                    obs_image_resolution=(640, 480),            # 图像分辨率1280, 720    # recording resolution
                    frequency=frequency,                        # 控制频率
                    init_joints=init_joints,                    # 初始化关节
                    enable_multi_cam_vis=True,                  # 多相机可视化
                    record_raw_video    =True,                  # 记录原始视频,number of threads per camera view for video recording (H.264)
                    thread_per_video    =3,                     # 每个相机视图的线程数用于视频录制
                    video_capture_resolution=(640, 480),        # 视频捕获分辨率
                    video_crf=21,                               # 视频录制质量，越低越好（但速度较慢）
                    multi_cam_vis_resolution=(640,480),         # 多相机可视化分辨率
                    shm_manager=shm_manager,                    # 共享内存管理器
                    enable_sam2 = True,                         # 是否启用SAM2
                ) as env:                                       # 实例化RealEnvFranka类
            cv2.setNumThreads(1)                                # 设置OpenCV线程数，可修改
            # 待研究
            # #设置RealSense曝光
            # env.realsense.set_exposure(exposure=90, gain=0) # 120__166
            # # 设置RealSense白平衡
            # env.realsense.set_white_balance(white_balance=4600) # 5900_4600

            time.sleep(1.0)                                     # 休眠1秒
            print(' ✅ Ready!')
            state = env.get_robot_state()                       # 获取机器人状态
            # target_pose = state['TargetTCPPose']
            target_pose = state['ActualTCPPose']                # 获取目标姿态
            gripper_state = state['ActualGripperstate']         # 获取夹爪状态
            print('Initial pose:', target_pose)                 # 打印初始姿态
            print('Initial gripper state:', gripper_state)      # 打印初始夹爪状态
            t_start = time.monotonic()                          # 获取当前时间
            iter_idx = 0                                        # 迭代次数 
            stop = False                                        # 停止标志 
            is_recording = False                                # 录制标志

            print_flag = True


            return_begin = False
            return_flag = False
            init_positon = np.array([0.3551,-0.0026,0.6935,-2.8691,1.2213,0.0139])
            epsilon = 0.02                                      # 位置误差容忍阈值 (单位：米)0.004
            step_size = 0.02                                    # 调整步长0.01
            epsilon_rot = 0.0175                                # 角度死区阈值 (约1度，单位：弧度)
            max_step_angle = 0.07                               # 最大单步调整角度 (约5度，单位：弧度)
                    
            stage = 0
            while not stop:
                # 1.计算时间
                t_cycle_end = t_start + (iter_idx + 1) * dt     # 计算循环结束时间
                t_sample = t_cycle_end - command_latency        # 计算采样时间
                t_command_target = t_cycle_end + dt             # 计算命令目标

                # 2.获取观察结果
                if print_flag:
                    while not env.is_ready:
                        print("Waiting for robot to be ready...")
                        time.sleep(1)     
                    print("Robot is ready!")   
                    print_flag = False                   
                obs = env.get_obs()

                # 3.处理按键事件
                #     如果按下的是'q'键，退出程序
                #     如果按下的是'c'键，开始录制
                #     如果按下的是's'键，停止录制
                #     如果按下的是退格键，# 删除最近录制的集
                press_events = key_counter.get_press_events()   # 获取按键事件
                for key_stroke in press_events:                 # 遍历按键事件
                    if key_stroke == KeyCode(char='q'):
                        env.end_episode()
                        key_counter.clear()                     # 清除按键计数器
                        stop = True                             # 设置停止标志为True
                        cv2.destroyAllWindows()                 # 关闭所有OpenCV窗口    
                    elif key_stroke == KeyCode(char='w'): 
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time()) # 开始新集
                        key_counter.clear()                     # 清除按键计数器
                        is_recording = True                     # 设置录制标志为True
                        return_begin = False
                        print('Recording!')                     # 打印录制消息
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()                       # 结束当前集
                        key_counter.clear()                     # 清除按键计数器
                        is_recording = False                    # 设置录制标志为False
                        return_begin = False
                        return_flag = True
                        print('Stopped.')                       # 打印停止消息
                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure to drop an episode?'): # 确认删除
                            env.drop_episode()                  # 删除集
                            key_counter.clear()                 # 清除按键计数器
                            is_recording = False                # 设置录制标志为False
                    elif key_stroke == KeyCode(char='r'):
                        return_begin = True
                        print("r被按下, Return to initial pose triggered")
                
                stage = key_counter[Key.space]                  # 获取空格键的按键次数
                gripper_state = (stage) % 2 *0.08

                gripper_now = obs['robot_gripper'][-1]                # 获取当前夹爪状态
                # print("gripper_state",gripper_state,"gripper_now",gripper_now[0])
                if gripper_now < gripper_state:
                    print("gripper is closing!")
                else:
                    print("gripper is opening!")
                
                # 4.可视化

                # 方案1：处理RGB图像数据（480x640x3）
                # vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()   # 获取可视化图像并转换颜色空间
            
                # 方案2：处理mask数据（480x640）
                mask = obs[f'camera_{vis_camera_idx}'][-1,:,:].copy()
                vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为3通道BGR图像
                vis_img = vis_img[:, :, ::-1].copy()  # 转为RGB格式（仅当显示库需要时）
                episode_id = env.replay_buffer.n_episodes                               # 获取当前集ID
                text = f'Episode: {episode_id}, Stage: {stage}'                         # 设置文本为当前集ID和阶段
                if is_recording:                                                        # 如果正在录制
                    text += ', Recording!'                                              # 添加录制文本
                cv2.putText(                                                            # 在可视化图像上绘制文本
                    vis_img,
                    text,
                    (10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255, 255, 255)
                )
                # 绘制第二行文本，y轴坐标调整一下使其位于第一行文本下方
                cv2.putText(
                    vis_img,
                    f'robot: {target_pose[:3]}',
                    (10, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255, 255, 255)
                )
                # 绘制第三行文本，y轴坐标调整一下使其位于第二行文本下方
                cv2.putText(
                    vis_img,
                    f'gripper_state: {gripper_state}',
                    (10, 90),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255, 255, 255)
                )
                # 处理RGB图像
                # cv2.imshow('Rgb Image', vis_img)                                      # 显示RGB图像
                cv2.imshow('Mask Image', vis_img)                                       # 显示RGB图像
                cv2.pollKey()                                                           # OpenCV键盘事件处理
                precise_wait(t_sample)
                
                if not env.is_saving:  # 仅在非保存状态响应空间鼠标
                    # 模拟空间鼠标
                    if return_begin:
                        # ===== 位置控制 =====
                        for i in range(3):
                            # 计算当前位置与目标的差值
                            error = init_positon[i] - target_pose[i]
                            abs_error = abs(error)
                            
                            if abs_error > epsilon:
                                # 差值超过阈值时，按比例调整步长
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
                        # ===== 旋转控制结束 =====
                    # 5.遥操作
                    # 5.1 获取遥操作命令
                    else:
                        sm_state = sm.get_motion_state_transformed()                    # 获取SpaceMouse的运动状态
                        dpos = sm_state[:3] * (env.max_pos_speed / frequency)           # 计算位置增量
                        drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)       # 计算旋转增量
                        # print("dpos:",dpos,"drot_xyz:",drot_xyz)
                        drot = st.Rotation.from_euler('xyz', drot_xyz)                  # 计算旋转
                        target_pose[:3] += dpos                                         # 更新目标位置
                        target_pose[3:6] = (drot * st.Rotation.from_rotvec(target_pose[3:6])).as_rotvec()# 更新目标旋转

                    if return_flag:
                        target_pose[3] = init_positon[3] 
                        target_pose[4] = init_positon[4] 
                        target_pose[5] = init_positon[5] 
                        gripper_state = 0.08
                        # 如果需要，可以设置move to start
                        return_flag = False
                else:
                    # 保存期间发送零动作保持机器人静止
                    print("Saving, sending zero action.")
                
                actions =np.zeros(7)
                actions[:6] = target_pose
                actions[6] = gripper_state
                # print("return_begin",return_begin)
                # 5.2 执行遥操作命令
                env.exec_actions(                                                       # 执行动作
                    actions=[actions], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[[stage]],
                    return_flag = [return_flag])
                # if return_flag:
                #     return_flag = False

                precise_wait(t_cycle_end)                                               # 精确等待循环结束时间
                iter_idx += 1                                                           # 增加迭代索引
            
            cv2.destroyAllWindows()                                                     # 关闭所有OpenCV窗口

# %%
if __name__ == '__main__':
    main()