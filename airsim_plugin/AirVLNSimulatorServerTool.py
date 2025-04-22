import argparse
import threading
import msgpackrpc
from pathlib import Path
import glob
import time
import os
import json
import sys
import subprocess
import errno
import signal
import copy


AIRSIM_SETTINGS_TEMPLATE = {
    "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
    "SettingsVersion": 1.2,
    "SimMode": "ComputerVision", # ComputerVision / Multirotor
    "ViewMode": "NoDisplay", # Fpv / NoDisplay
    "ClockSpeed": 1,
    # "LocalHostIp": "127.0.0.1",
    # "ApiServerPort": 10000,
    "CameraDefaults": {
        "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": 224,
                "Height": 224,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03
            },
            {
                "ImageType": 2,
                "Width": 256,
                "Height": 256,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03
            },
            {
                "ImageType": 3,
                "Width": 256,
                "Height": 256,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03
            }
        ],
        "X": 0,
        "Y": 0,
        "Z": 0,
        "Pitch": 0,
        "Roll": 0,
        "Yaw": 0
    },
    "Recording": {
        "RecordInterval": 0.001,
        "Enabled": False,
        "Cameras": []
    },
    "SubWindows": [],
    "Vehicles": {}
}


def create_drones(drone_num_per_env=1, show_scene=False, uav_mode=False) -> dict:
    airsim_settings = copy.deepcopy(AIRSIM_SETTINGS_TEMPLATE)

    if show_scene == True:
        airsim_settings['ViewMode'] = 'Fpv'
    else:
        airsim_settings['ViewMode'] = 'NoDisplay'

    if uav_mode == True:
        airsim_settings['SimMode'] = 'Multirotor'
        airsim_settings['PhysicsEngineName'] = 'ExternalPhysicsEngine'
    else:
        airsim_settings['SimMode'] = 'ComputerVision'


    # create drone objects
    for i in range(drone_num_per_env):
        drone_name = 'Drone_' + str(i+1)

        airsim_settings['Vehicles'][str(drone_name)] = {}

        drone = {
            "VehicleType": "ComputerVision",
            "Cameras": {
                "front_0": {
                    "CaptureSettings": [
                        {
                            "ImageType": 0,
                            "Width": 224,
                            "Height": 224,
                            "FOV_Degrees": 90,
                            "AutoExposureMaxBrightness": 1,
                            "AutoExposureMinBrightness": 0.03
                        },
                        {
                            "ImageType": 2,
                            "Width": 256,
                            "Height": 256,
                            "FOV_Degrees": 90,
                            "AutoExposureMaxBrightness": 1,
                            "AutoExposureMinBrightness": 0.03
                        },
                        {
                            "ImageType": 3,
                            "Width": 256,
                            "Height": 256,
                            "FOV_Degrees": 90,
                            "AutoExposureMaxBrightness": 1,
                            "AutoExposureMinBrightness": 0.03
                        }
                    ],
                    "X": 0.5, "Y": 0, "Z": 0,
                    "Pitch": 0, "Roll": 0, "Yaw": 0
                }
            },
            "X": 0, "Y": 0, "Z": 0,
            "Pitch": 0, "Roll": 0, "Yaw": 0
        }

        if airsim_settings['SimMode'] == 'ComputerVision':
            drone['VehicleType'] = 'ComputerVision'
        elif airsim_settings['SimMode'] == 'Multirotor':
            drone['VehicleType'] = 'SimpleFlight'
        else:
            raise NotImplementedError

        airsim_settings['Vehicles'][str(drone_name)] = copy.deepcopy(drone)

    return airsim_settings


def pid_exists(pid) -> bool:
    """
    Check whether pid exists in the current process table.
    UNIX only.
    """
    if pid < 0:
        return False

    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True


def FromPortGetPid(port: int):
    subprocess_execute = "netstat -nlp | grep {}".format(
        port,
    )

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'FromPortGetPid',
                e,
            )
        )
        return None
    except:
        return None

    pid = None
    for line in iter(p.stdout.readline, b''):
        line = str(line, encoding="utf-8")
        if 'tcp' in line:
            pid = line.strip().split()[-1].split('/')[0]
            try:
                pid = int(pid)
            except:
                pid = None
            break

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    return pid


def KillPid(pid) -> None:
    if pid is None or not isinstance(pid, int):
        print('pid is not int')
        return

    while pid_exists(pid):
        try:
            # os.system(("kill -9 {}".format(pid)))
            os.kill(pid, signal.SIGKILL)
        except Exception as e:
            pass
        time.sleep(0.5)

    return


def KillPorts(ports) -> None:
    threads = []

    def _kill_port(index, port):
        pid = FromPortGetPid(port)
        KillPid(pid)

    for index, port in enumerate(ports):
        thread = threading.Thread(target=_kill_port, args=(index, port))
        threads.append(thread)
    for thread in threads:
        thread.setDaemon(True)
        thread.start()
    for thread in threads:
        thread.join()
    threads = []

    return


def KillAirVLN() -> None:
    subprocess_execute = "pkill -9 AirVLN"

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'KillAirVLN',
                e,
            )
        )
        return
    except:
        return

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    time.sleep(1)
    return


class EventHandler(object):
    def __init__(self):
        scene_ports = []
        for i in range(1000):
            scene_ports.append(
                int(args.port) + (i+1)
            )
        self.scene_ports = scene_ports

        scene_gpus = []
        while len(scene_gpus) < 100:
            scene_gpus += GPU_IDS.copy()
        self.scene_gpus = scene_gpus

        self.scene_used_ports = []

    def ping(self) -> bool:
        return True

    def _open_scenes(self, ip: str , scen_ids: list):
        print(
            "{}\tSTART closing scenes ".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        KillPorts(self.scene_used_ports)
        self.scene_used_ports = []
        # KillAirVLN()
        print(
            "{}\tEND closing scenes ".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )


        # Occupied airsim port 1
        ports = []
        index = 0
        while len(ports) < len(scen_ids):
            pid = FromPortGetPid(self.scene_ports[index])
            if pid is None or not isinstance(pid, int):
                ports.append(self.scene_ports[index])
            index += 1

        KillPorts(ports)


        # Occupied GPU 2
        gpus = [self.scene_gpus[index] for index in range(len(scen_ids))]


        # search scene path 3
        choose_env_exe_paths = []
        for scen_id in scen_ids:
            if str(scen_id).lower() == 'none':
                choose_env_exe_paths.append(None)
                continue

            res = glob.glob((str(SEARCH_ENVs_PATH) + '/**/' + 'env_' + str(scen_id) + '/LinuxNoEditor/AirVLN.sh'), recursive=True)
            if len(res) > 0:
                choose_env_exe_paths.append(res[0])
            else:
                print(f'can not find scene file: {scen_id}')
                raise KeyError


        p_s = []
        for index in range(len(scen_ids)):
            # airsim settings 4
            airsim_settings = create_drones()
            airsim_settings['ApiServerPort'] = int(ports[index])
            airsim_settings_write_content = json.dumps(airsim_settings)
            if not os.path.exists(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1))):
                os.makedirs(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1)), exist_ok=True)
            with open(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1) / 'settings.json'), 'w', encoding='utf-8') as dump_f:
                dump_f.write(airsim_settings_write_content)


            # open scene 5
            if choose_env_exe_paths[index] is None:
                p_s.append(None)
                continue
            else:
                subprocess_execute = "bash {} -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter={} --settings {} ".format(
                    choose_env_exe_paths[index],
                    gpus[index],
                    str(CWD_DIR / 'airsim_plugin/settings' / str(index+1) / 'settings.json'),
                )

                try:
                    p = subprocess.Popen(
                        subprocess_execute,
                        stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        shell=True,
                    )
                    p_s.append(p)
                except Exception as e:
                    print(
                        "{}\t{}".format(
                            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                            e,
                        )
                    )
                    return False, None
                except:
                    return False, None
        time.sleep(3)

        # check
        threads = []

        def _check_scene(index, p):
            if p is None:
                print(
                    "{}\tOpening {}-th scene (scene {})\tgpu:{}".format(
                        str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                        index,
                        None,
                        gpus[index],
                    )
                )
                return

            for line in iter(p.stdout.readline, b''):
                if 'Drone_' in str(line):
                    break

            try:
                p.terminate()
                # os.system(("kill -9 {}".format(p.pid)))
                os.kill(p.pid, signal.SIGKILL)
            except:
                pass

            print(
                "{}\tOpening {}-th scene (scene {})\tgpu:{}".format(
                    str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                    index,
                    scen_ids[index],
                    gpus[index],
                )
            )
            return

        for index, p in enumerate(p_s):
            thread = threading.Thread(target=_check_scene, args=(index, p))
            threads.append(thread)
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        threads = []

        # ChangeNice(ports)

        self.scene_used_ports += copy.deepcopy(ports)

        return True, (ip, ports)

    def reopen_scenes(self, ip: str, scen_ids: list):
        print(
            "{}\tSTART reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        try:
            result = self._open_scenes(ip, scen_ids)
        except Exception as e:
            print(e)
            result = False, None
        print(
            "{}\tEND reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result

    def close_scenes(self, ip: str) -> bool:
        print(
            "{}\tSTART close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )

        try:
            KillPorts(self.scene_used_ports)
            self.scene_used_ports = []
            # KillPorts(self.scene_ports)
            # KillAirVLN()

            result = True
        except Exception as e:
            print(e)
            result = False

        print(
            "{}\tEND close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result


def serve_background(server, daemon=False):
    def _start_server(server):
        server.start()
        server.close()

    t = threading.Thread(target=_start_server, args=(server,))
    t.setDaemon(daemon)
    t.start()
    return t


def serve(daemon=False):
    try:
        server = msgpackrpc.Server(EventHandler())
        addr = msgpackrpc.Address(HOST, PORT)
        server.listen(addr)

        thread = serve_background(server, daemon)

        return addr, server, thread
    except Exception as err:
        print(err)
        pass


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        default='0',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help='server port'
    )
    args = parser.parse_args()


    HOST = '127.0.0.1'
    PORT = int(args.port)

    CWD_DIR = Path(str(os.getcwd())).resolve()
    PROJECT_ROOT_DIR = CWD_DIR.parent
    SEARCH_ENVs_PATH = PROJECT_ROOT_DIR / 'ENVs'
    assert os.path.exists(str(SEARCH_ENVs_PATH)), 'error'

    gpu_list = []
    gpus = str(args.gpus).split(',')
    for gpu in gpus:
        gpu_list.append(int(gpu.strip()))
    GPU_IDS = gpu_list.copy()


    addr, server, thread = serve()
    print(f"start listening \t{addr._host}:{addr._port}")

