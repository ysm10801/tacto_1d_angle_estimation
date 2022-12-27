# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import cv2
import pybullet as p
import pybulletX as px
import tacto  # Import TACTO
import sys
import os
import time
import random

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from tacto.sensorPosEst import SensorPosEst


log = logging.getLogger(__name__)

# Load the config YAML file from examples/conf/digit.yaml
def main(objZorn):
    # Initialize digits
    bg = cv2.imread("conf/bg_digit_240_320.jpg")
    digits = tacto.Sensor(width = 120, height = 160, visualize_gui = False, background=bg)
    # digits = SensorPosEst(**cfg.tacto, background=bg)


    # Initialize World
    log.info("Initializing world")
    px.init(mode=p.DIRECT)
    # px.init()

    p.resetDebugVisualizerCamera(cameraDistance = 0.12,
                                 cameraYaw = 90. ,
                                 cameraPitch = -45. ,
                                 cameraTargetPosition = [0, 0, 0])

    # Create and initialize DIGIT
    # digit_body = px.Body(**cfg.digit)
    digit_body = px.Body("../meshes/digit.urdf",
                        base_position = [0, 0, 0],
                        base_orientation = [0.0, -0.707106, 0.0, 0.707106],
                        use_fixed_base = True)
    digits.add_camera(digit_body.id, [-1])

    # objXpos=random.uniform(-0.017, -0.012)
    objXpos = -0.015
    # objYpos=random.uniform(-0.005, 0.005)
    objYpos = 0
    objZorn_ = p.getQuaternionFromEuler([0.0, 0.0, objZorn])

    # Add object to pybullet and tacto simulator
    obj = px.Body("objects/cube_small_PosEst.urdf",
                    base_position = [objXpos, objYpos, 0.021],
                    base_orientation = objZorn_,
                    global_scaling = 0.1,
                    use_fixed_base = False
                    )
    digits.add_body(obj)

    # Create control panel to control the 6DoF pose of the object
    # panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
    # panel.start()
    # log.info("Use the slides to move the object until in contact with the DIGIT")

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    CylPos, CylOrn = p.getBasePositionAndOrientation(obj.id)
    p.applyExternalForce(objectUniqueId=obj.id, linkIndex=-1,
                            forceObj=[0,0,-0.5], posObj=[CylPos[0],CylPos[1],0.023], flags=p.WORLD_FRAME)
    t.start()

    for j in range (duration):
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        time.sleep(1./240.)
        if j == duration-1 :
            colors = np.concatenate(color, axis=1)
            b, g, r = cv2.split(colors)
            img = cv2.merge([r,g,b])
            filename = "/home/yang/tacto/examples/data1/"+str(i+1).zfill(4)+"_"+str(objZorn)+".png"
            cv2.imwrite(filename, img)
    # colorbg = cv2.imread("conf/bg_digit_240_320.jpg")

    # for j in range (duration):
    #     colorz = np.zeros_like(color)
    #     depthz = np.zeros_like(depth)
    #     digits.updateGUI(colorz, depthz)
    #     time.sleep(1./240.)
    #     j+=1
    #     if j == duration-1 :
    #         pass


    # p.removeBody(obj.id)


dat_num = 31415
duration = 10
for i in range(dat_num):
    if __name__ == "__main__":
        angle = round((i+1) * 0.0001, 4)
        main(angle)
        p.disconnect()
        # p.resetSimulation()