import msgpackrpc
import time
import airsim
import threading
import random
import copy
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import sys
    cur_path=os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, cur_path+"/..")
    print(os.getcwd())
    from src.common.param import args
else:
    from src.common.param import args

from utils.logger import logger


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        try:
            self.result = self.func(*self.args)
        except Exception as e:
            logger.error(e)
            self.flag_ok = False
        else:
            self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:
    def __init__(self, machines_info) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info ]

        self._init_check()

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self, socket_client: msgpackrpc.Client) -> bool:
        try:
            socket_client.call('ping')
            logger.info("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                logger.error("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                logger.error('Ping returned false')
            return False

    def _confirmConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    self.airsim_clients[index_1][index_2].confirmConnection()

        return

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients

        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass

        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass

        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    def run_call(self, airsim_timeout: int=60) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=180)
            )

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients


        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            logger.info(f'Failed to open scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
            result = socket_client.call('reopen_scenes', socket_client.address._host, self.machines_info[index]['open_scenes'])

            if result[0] == False:
                logger.error(f'Failed to open scenes, machine : {socket_client.address._host}:{socket_client.address._port}')
                raise Exception('Failed to open scenes')
            assert len(result[1]) == 2, 'Failed to open scenes'

            ip = result[1][0]
            ports = result[1][1]

            if isinstance(ip, bytes):
                ip = ip.decode()

            assert str(ip) == str(socket_client.address._host), 'Failed to open scenes'
            assert len(ports) == len(self.machines_info[index]['open_scenes']), 'Failed to open scenes'
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:
                    self.airsim_clients[index][i] = airsim.VehicleClient(ip=ip, port=port, timeout_value=airsim_timeout)

            logger.info(f'Failed to open scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
            return

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                MyThread(_run_command, (index, socket_client))
            )
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            raise Exception('Failed to open scenes')

        after = time.time()
        diff = after - before
        logger.info(f"Start time: {diff}")

        self._confirmConnection()
        self._closeSocketConnection()

    def getImageResponses(self, get_rgb=True, get_depth=True):

        def _getImages(airsim_client: airsim.VehicleClient, scen_id, get_rgb, get_depth):
            if airsim_client is None:
                raise Exception('error')
                return None, None

            img_rgb = None
            img_depth = None

            if not get_rgb and not get_depth:
                return None, None

            if scen_id in [1, 7]:
                time_sleep_cnt = 0
                while True:
                    try:
                        ImageRequest = []
                        if get_rgb:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                            )
                        if get_depth:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.DepthVis, pixels_as_float=False, compress=True)
                            )

                        responses = airsim_client.simGetImages(ImageRequest, vehicle_name='Drone_1')

                        if get_rgb and get_depth:
                            response_rgb = responses[0]
                            response_depth = responses[1]
                        elif get_rgb and not get_depth:
                            response_rgb = responses[0]
                        elif not get_rgb and get_depth:
                            response_depth = responses[0]
                        else:
                            break


                        img_rgb = None
                        img_depth = None

                        if get_rgb:
                            assert response_rgb.height == args.Image_Height_RGB and response_rgb.width == args.Image_Width_RGB, 'Failed to retrieve RGB image'

                            img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
                            if args.run_type not in ['eval']:
                                assert not (img1d.flatten()[0] == img1d).all(), 'Failed to retrieve RGB image'
                            img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
                            img_rgb = np.array(img_rgb)

                        if get_depth:
                            assert response_depth.height == args.Image_Height_DEPTH and response_depth.width == args.Image_Width_DEPTH, 'Failed to retrieve DEPTH image'

                            png_file_name = '/tmp/AirVLN_depth_{}_{}.png'.format(time.time(), random.randint(0, 10000))
                            airsim.write_file(png_file_name, response_depth.image_data_uint8)
                            img3d = cv2.imread(png_file_name)

                            os.remove(png_file_name)

                            img1d = img3d[:, :, 1]
                            img1d = img1d.reshape(response_depth.height, response_depth.width, 1)

                            obs_depth_img = img1d / 255

                            img_depth = np.array(obs_depth_img, dtype=np.float32)

                        break
                    except:
                        time_sleep_cnt += 1
                        logger.error("Image retrieval error")
                        logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                        time.sleep(1)

                    if time_sleep_cnt > 20:
                        raise Exception('Failed to retrieve image')

            else:
                time_sleep_cnt = 0
                while True:
                    try:
                        ImageRequest = []
                        if get_rgb:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                            )
                        if get_depth:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
                            )

                        responses = airsim_client.simGetImages(ImageRequest, vehicle_name='Drone_1')

                        if get_rgb and get_depth:
                            response_rgb = responses[0]
                            response_depth = responses[1]
                        elif get_rgb and not get_depth:
                            response_rgb = responses[0]
                        elif not get_rgb and get_depth:
                            response_depth = responses[0]
                        else:
                            break

                        if get_rgb:
                            assert response_rgb.height == args.Image_Height_RGB and response_rgb.width == args.Image_Width_RGB, 'Failed to retrieve RGB image'

                            img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
                            img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
                            img_rgb = np.array(img_rgb)

                        if get_depth:
                            assert response_depth.height == args.Image_Height_DEPTH and response_depth.width == args.Image_Width_DEPTH, 'Failed to retrieve DEPTH image'

                            depth_img_in_meters = airsim.list_to_2d_float_array(response_depth.image_data_float, response_depth.width, response_depth.height)
                            if depth_img_in_meters.min() < 1e4:
                                assert not (depth_img_in_meters.flatten()[0] == depth_img_in_meters).all(), 'Failed to retrieve DEPTH image'
                            depth_img_in_meters = depth_img_in_meters.reshape(response_depth.height, response_depth.width, 1)

                            obs_depth_img = np.clip(depth_img_in_meters, 0, 100)
                            obs_depth_img = obs_depth_img / 100

                            img_depth = np.array(obs_depth_img, dtype=np.float32)

                        break
                    except:
                        time_sleep_cnt += 1
                        logger.error("Failed to retrieve image")
                        logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                        time.sleep(1)

                    if time_sleep_cnt > 20:
                        raise Exception('Failed to retrieve image')

            # Tip: Before using AirVLN code, please confirm that the channel order 
            #       of the images captured is as expected by visualization!
            # Example is as below:

            # plt.imsave('./tmp/img_rgb.png', img_rgb)

            # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

            # plt.imsave('./tmp/img_rgb_converted.png', img_rgb)

            # plt.imsave('./tmp/img_depth.png', img_depth.squeeze(), cmap='gray')

            return img_rgb, img_depth

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], self.machines_info[index_1]['open_scenes'][index_2], get_rgb, get_depth))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses failed')
            return None

        return responses

    def setPoses(self, poses: list) -> bool:
        def _setPoses(airsim_client: airsim.VehicleClient, pose: airsim.Pose) -> None:
            if airsim_client is None:
                raise Exception('error')
                return

            airsim_client.simSetVehiclePose(
                pose=pose,
                ignore_collision=True,
                vehicle_name='Drone_1',
            )

            return

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_setPoses, (self.airsim_clients[index_1][index_2], poses[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('setPoses failed')
            return False

        return True

    def closeScenes(self):
        try:
            socket_clients = []
            for index, item in enumerate(self.machines_info):
                socket_clients.append(
                    msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=180)
                )

            for socket_client in socket_clients:
                if not self._confirmSocketConnection(socket_client):
                    logger.error('cannot establish socket')
                    raise Exception('cannot establish socket')

            self.socket_clients = socket_clients


            self._closeConnection()

            def _run_command(index, socket_client: msgpackrpc.Client):
                logger.info(f'START closing all scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
                result = socket_client.call('close_scenes', socket_client.address._host)
                logger.info(f'END closing all scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
                return

            threads = []
            for index, socket_client in enumerate(socket_clients):
                threads.append(
                    MyThread(_run_command, (index, socket_client))
                )
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
            threads = []

            self._closeSocketConnection()
        except Exception as e:
            logger.error(e)


if __name__ == '__main__':

    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 1,
            'open_scenes': [1],
        },
    ]
    # machines_info_xxx = [
    #     {
    #         'MACHINE_IP': '127.0.0.1',
    #         'SOCKET_PORT': 30000,
    #         'MAX_SCENE_NUM': 8,
    #         'open_scenes': [1, 2, 3, 4, 5, 6, 7, None],
    #     },
    # ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    start_time = time.time()
    while True:
        time_1 = time.time()
        responses = tool.getImageResponses()
        time_2 = time.time()
        print(
            "total_time: {} \t time: {} \t fps: {}".format(
                (time_2-start_time),
                (time_2-time_1),
                1/(time_2-time_1),
            )
        )

        poses = []
        for index_1, item in enumerate(machines_info_xxx):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                pose=airsim.Pose(
                    position_val=airsim.Vector3r(random.randint(0, 1000), random.randint(0, 1000), random.randint(-200, 0)),
                    orientation_val=airsim.Quaternionr(0, 0, 0, 1),
                )
                poses[index_1].append(pose)

        tool.setPoses(poses)

