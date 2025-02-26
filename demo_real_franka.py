"""
使用方法：
(robodiff)$ python demo_real_robot.py -o <演示保存目录> --robot_ip <ur5的ip地址>
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

机器人移动：
移动您的SpaceMouse以移动机器人的末端执行器（锁定在xy平面）。
按下SpaceMouse右键以解锁z轴。
按下SpaceMouse左键以启用旋转轴。

录制控制：
点击opencv窗口（确保它是焦点）。
按"C"开始录制。
按"S"停止录制。
按"Q"退出程序。
按"退格键"删除先前录制的剧集。
"""

# %%
import time
from multiprocessing.managers               import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform              as st
from real_world.real_env                    import RealEnv
from real_world.spacemouse_shared_memory    import Spacemouse
from real_world.precise_sleep               import precise_wait
from real_world.keystroke_counter           import ( KeystrokeCounter, Key, KeyCode  )

import open3d as o3d

@click.command()
@click.option('--output', '-o', required=True, help="保存演示数据集的目录Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5的IP地址,例如192.168.0.204 UR5's IP address e.g. 192.168.0.204")
@click.option('--vis_camera_idx', default=0, type=int, help="要可视化的RealSense摄像头索引Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="是否在开始时初始化机器人的关节配置Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="控制频率,以赫兹为单位Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="接收到SpaceMouse命令到机器人执行之间的延迟,以秒为单位 Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency): # 定义主函数，接收五个参数：输出路径、机器人IP、可视化相机索引、初始化关节、频率和命令延迟
    dt = 1/frequency                    # 计算时间步长
    # 使用共享内存管理器、按键计数器、SpaceMouse、真实环境
    with SharedMemoryManager() as shm_manager: 
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager) as sm, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                obs_image_resolution=(1280,720), # 记录分辨率
                frequency=frequency, 
                init_joints=init_joints, 
                enable_multi_cam_vis=True, 
                record_raw_video=True, 
                thread_per_video=3,     # 每个相机视图的视频录制线程数
                video_crf=21,           # 视频录制质量，值越低质量越好（但速度较慢）
                shm_manager=shm_manager
            ) as env:                   # 初始化真实环境
            cv2.setNumThreads(1)        # 设置OpenCV线程数为1
            # 待研究
            # #设置RealSense曝光
            # env.realsense.set_exposure(exposure=90, gain=0) # 120__166
            # # 设置RealSense白平衡
            # env.realsense.set_white_balance(white_balance=4600) # 5900_4600
            time.sleep(1.0)                         # 等待1秒
            print('Ready!')                         # 打印准备就绪消息
            state = env.get_robot_state()           # 获取机器人状态
            target_pose = state['TargetTCPPose']    # 获取目标姿势
            t_start = time.monotonic()              # 获取当前时间
            iter_idx = 0                            # 初始化迭代索引为0
            stop = False                            # 初始化停止标志为False
            is_recording = False                    # 初始化录制标志为False
            while not stop:
                # 1.计算时间
                t_cycle_end = t_start + (iter_idx + 1) * dt     # 计算循环结束时间
                t_sample = t_cycle_end - command_latency        # 计算采样时间
                t_command_target = t_cycle_end + dt             # 计算命令目标时间
                # 2.获取观察结果
                obs = env.get_obs() 
                # 3.处理按键事件
                #     如果按下的是'q'键，退出程序
                #     如果按下的是'c'键，开始录制
                #     如果按下的是's'键，停止录制
                #     如果按下的是退格键，# 删除最近录制的集
                press_events = key_counter.get_press_events()   # 获取按键事件
                for key_stroke in press_events:                 # 遍历按键事件
                    if key_stroke == KeyCode(char='q'):
                        stop = True                             # 设置停止标志为True
                    elif key_stroke == KeyCode(char='c'): 
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time()) # 开始新集
                        key_counter.clear()                     # 清除按键计数器
                        is_recording = True                     # 设置录制标志为True
                        print('Recording!')                     # 打印录制消息
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()                       # 结束当前集
                        key_counter.clear()                     # 清除按键计数器
                        is_recording = False                    # 设置录制标志为False
                        print('Stopped.')                       # 打印停止消息
                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure to drop an episode?'): # 确认删除
                            env.drop_episode()                  # 删除集
                            key_counter.clear()                 # 清除按键计数器
                            is_recording = False                # 设置录制标志为False
                stage = key_counter[Key.space]                  # 获取空格键的按键次数

                # 4.可视化
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy() # 获取可视化图像并转换颜色空间
                depth_img = obs[f'camera_{vis_camera_idx}_depth'][-1].copy()  # 获取深度图像数据
                episode_id = env.replay_buffer.n_episodes       # 获取当前集ID
                # 信息1
                text = f'Episode: {episode_id}, Stage: {stage}' # 设置文本为当前集ID和阶段
                if is_recording:                                # 如果正在录制
                    text += ', Recording!'                      # 添加录制文本
                cv2.putText(                                    # 在可视化图像上绘制文本
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )
                # 绘制第二行文本，y轴坐标调整一下使其位于第一行文本下方
                text_now = f'robot: {target_pose[:3]}, gripper: {target_pose[6]}'
                cv2.putText(
                    vis_img,
                    text_now,
                    (10, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255, 255, 255)
                )
                
                
                cv2.imshow('Rgb Image', vis_img)                  # 显示可视化图像
                cv2.imshow('Depth Image', depth_img)
                
                # 获取点云数据，这里假设你有点云数据保存在 `obs['pointcloud']`
                # 如果没有，你需要根据你的实际数据来源进行修改
                # point_cloud = obs.get('pointcloud', None)  # 从环境中获取点云数据
                # if point_cloud is not None:
                #     pc = o3d.geometry.PointCloud()# 将点云转换为open3d格式，假设点云是一个Nx3的数组
                #     pc.points = o3d.utility.Vector3dVector(point_cloud)
                #     o3d.visualization.draw_geometries([pc], window_name="Point Cloud Visualization")# 可视化点云
                
                cv2.pollKey()                                   # 处理按键事件
                precise_wait(t_sample)                          # 精确等待采样时间
                
                
                
                
                
                # 5.遥操作
                # 5.1 获取遥操作命令
                sm_state = sm.get_motion_state_transformed()    # 获取SpaceMouse的运动状态
                # print(sm_state)                               # 打印运动状态
                dpos = sm_state[:3] * (env.max_pos_speed / frequency)       # 计算位置增量
                drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)   # 计算旋转增量

                # 0没按下，拒绝旋转
                if not sm.is_button_pressed(0):
                    drot_xyz[:] = 0                             # 旋转增量置零
                else:
                    dpos[:] = 0                                 # 位置增量置零

                # 1没按下，拒绝上下位移
                if not sm.is_button_pressed(1): 
                    dpos[2] = 0                                 # Z轴位置增量置零    
                drot = st.Rotation.from_euler('xyz', drot_xyz)  # 计算旋转
                target_pose[:3] += dpos                         # 更新目标位置
                target_pose[3:6] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:6])).as_rotvec()               # 更新目标旋转
                
                if sm.is_button_pressed(12):                    # close gripper
                    # dpos = -gripper_speed / frequency
                    target_pose[6] = True
                    print("sm_state[1] is down")  
                # 如果SpaceMouse的第二个按钮被按下，则设置夹爪的平移速度为正值，表示夹爪打开
                elif sm.is_button_pressed(13):
                    target_pose[6] = False
                    print("sm_state[2] is down")  
                    # dpos = gripper_speed / frequency
                # 绘制第二行文本，y轴坐标调整一下使其位于第一行文本下方
                text_tar = f'robot_tar: {target_pose[:3]}, gripper_tar: {target_pose[6]}'
                cv2.putText(
                    vis_img,
                    text_tar,
                    (10, 90),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255, 255, 255)
                )
                # 5.2 执行遥操作命令
                env.exec_actions(                               # 执行动作
                    actions=[target_pose], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)                       # 精确等待循环结束时间
                iter_idx += 1                                   # 增加迭代索引


# %%
if __name__ == '__main__':
    main()
