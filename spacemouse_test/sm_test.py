from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
import scipy.spatial.transform      as st
from spacemouse_shared_memory       import Spacemouse
from multiprocessing.managers       import SharedMemoryManager


with SharedMemoryManager() as shm_manager:
    with    Spacemouse(shm_manager=shm_manager) as sm:
        while True:
            # 5.1 获取遥操作命令
            sm_state = sm.get_motion_state_transformed()        # 获取SpaceMouse的运动状态
            # print(sm_state)
            # print("sm_state:{}".format(sm_state))               # 打印运动状态
            # dpos = sm_state[:3] * (0.5 / frequency)
            # drot_xyz = sm_state[3:] * (1.5 / frequency)
            dpos = sm_state[:3] * (0.1 / sm.frequency)       # 计算位置增量
            drot_xyz = sm_state[3:] * (0.2 / sm.frequency)   # 计算旋转增量
            # print("max_pos_speed:{},max_rot_speed:{}".format(env.max_pos_speed,env.max_rot_speed))           # 打印max_speed

            # 旋转轴和Z轴解锁
            # 如果没有按下第一个按钮，平移模式
            drot = st.Rotation.from_euler('xyz', drot_xyz)      # 将旋转速度转换为一个旋转对象
            # 更新目标姿态，根据SpaceMouse的输入和当前控制机器人列表
            # target_pose[:3] += dpos
            # target_pose[3:] = (drot * st.Rotation.from_rotvec(
            #     target_pose[3:])).as_rotvec()
            dpos = 0                                            # 重置平移速度变量
            # 如果SpaceMouse的第一个按钮被按下，则设置夹爪的平移速度为负值，表示夹爪关闭
            # 0 menu
            # 23 alt
            if sm.is_button_pressed(12):                     # close gripper
                # dpos = -gripper_speed / frequency
                print("sm_state[1] is down")  
            # 如果SpaceMouse的第二个按钮被按下，则设置夹爪的平移速度为正值，表示夹爪打开
            if sm.is_button_pressed(13):
                print("sm_state[2] is down")  
                # dpos = gripper_speed / frequency


# class Spacemouse(mp.Process):
#     def __init__(self, 
#             frequency=200,
#             max_value=500, 
#             deadzone=(0,0,0,0,0,0), 
#             dtype=np.float32,
#             n_buttons=16,
#             ):
#         """
#         Continuously listen to 3D connection space navigator events
#         and update the latest state.

#         max_value: {300, 500} 300 for wired version and 500 for wireless
#         deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
#         """
#         super().__init__()
#         if np.issubdtype(type(deadzone), np.number):
#             deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
#         else:
#             deadzone = np.array(deadzone, dtype=dtype)
#         assert (deadzone >= 0).all()

#         # copied variables
#         self.frequency = frequency
#         self.max_value = max_value
#         self.dtype = dtype
#         self.deadzone = deadzone
#         self.n_buttons = n_buttons
#         self.tx_zup_spnav = np.array([
#             [0,0,-1],
#             [1,0,0],
#             [0,1,0]
#         ], dtype=dtype)

#         # shared variables
#         self.ready_event = mp.Event()
#         self.stop_event = mp.Event()

#         # Motion and button state placeholders
#         self.motion_event = np.zeros((7,), dtype=np.int64)
#         self.button_state = np.zeros((n_buttons,), dtype=bool)

#     # ======= get state APIs ==========

#     def get_motion_state(self):
#         state = np.array(self.motion_event[:6], dtype=self.dtype) / self.max_value
#         is_dead = (-self.deadzone < state) & (state < self.deadzone)
#         state[is_dead] = 0
#         return state
    
#     def get_motion_state_transformed(self):
#         """
#         Return in right-handed coordinate
#         z
#         *------>y right
#         |   _
#         |  (O) space mouse
#         v
#         x
#         back
#         """
#         state = self.get_motion_state()
#         tf_state = np.zeros_like(state)
#         tf_state[:3] = self.tx_zup_spnav @ state[:3]
#         tf_state[3:] = self.tx_zup_spnav @ state[3:]
#         return tf_state

#     def get_button_state(self):
#         return self.button_state
    
#     def is_button_pressed(self, button_id):
#         return self.get_button_state()[button_id]
    
#     #========== start stop API ===========

#     def start(self, wait=True):
#         super().start()
#         if wait:
#             self.ready_event.wait()
    
#     def stop(self, wait=True):
#         self.stop_event.set()
#         if wait:
#             self.join()
    
#     def __enter__(self):
#         self.start()
#         return self
    
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.stop()

#     # ========= main loop ==========
#     def run(self):
#         spnav_open()
#         try:
#             # send one message immediately so client can start reading
#             self.ready_event.set()

#             # while not self.stop_event.is_set():
#             #     event = spnav_poll_event()
#             #     if event is not None:
#             #         if isinstance(event, SpnavMotionEvent):
#             #             self.motion_event[:3] = event.translation
#             #             self.motion_event[3:6] = event.rotation
#             #             self.motion_event[6] = event.period
#             #         elif isinstance(event, SpnavButtonEvent):
#             #             self.button_state[event.bnum] = event.press
#             #     time.sleep(1/self.frequency)
#             while True:
#                 event = spnav.poll()
#                 if event is not None:
#                     if event.type == spnav.EVENT_BUTTON:
#                         button = event.button
#                         state = event.press
#                         print(f"Button event: {button}, state: {state}")
#                         if button < len(self.button_state):
#                             self.button_state[button] = state
#                         else:
#                             print(f"Button index out of range: {button}")

#         finally:
#             spnav_close()


# if __name__ == '__main__':
#     sm = Spacemouse(frequency=1000)# 根据机器人频率，选取一个具体的frequency
#     sm.run()
#     # with sm:   # 不再需要传入 shm_manager
#     #     while True:  # 假设这是在一个循环中持续获取状态
#     #         # 5.1 获取遥操作命令
#     #         sm_state = sm.get_motion_state_transformed()        # 获取SpaceMouse的运动状态
#     #         # print("sm_state:{}".format(sm_state))               # 打印运动状态
            
#     #         # 计算位置和旋转增量
#     #         dpos = sm_state[:3] * (0.1 / sm.frequency)       
#     #         drot_xyz = sm_state[3:] * (0.2 / sm.frequency)
            
#     #         # 旋转轴和Z轴解锁
#     #         drot = st.Rotation.from_euler('xyz', drot_xyz)  # 将旋转速度转换为一个旋转对象
            
#     #         # 更新目标姿态，根据SpaceMouse的输入
#     #         # target_pose[:3] += dpos
#     #         # target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()

#     #         dpos = 0  # 重置平移速度变量
            
#     #         # 判断按钮状态，更新夹爪位置
#     #         if sm.is_button_pressed(11):
#     #             print("sm_state[1] is down")  
#     #         if sm.is_button_pressed(12):
#     #             print("sm_state[0]:",sm.is_button_pressed(12))  
#     #         # if sm.is_button_pressed(0):  # 如果按下第一个按钮，关闭夹爪
#     #             # dpos = -gripper_speed / frequency
#     #         # if sm.is_button_pressed(1):  # 如果按下第二个按钮，打开夹爪
#     #             # dpos = gripper_speed / frequency

#     #         # 更新夹爪目标位置，确保夹爪的位置在0到最大夹爪宽度之间
#     #         # gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)



# 5.遥操作

