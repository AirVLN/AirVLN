import os
import sys
from pathlib import Path
sys.path.append(str(Path(str(os.getcwd())).resolve()))
import gc
import time
import lmdb
import tqdm
import math
import random
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from typing import List, Optional, DefaultDict
import msgpack_numpy

from utils.logger import logger
from utils.utils import get_rank, is_dist_avail_and_initialized, is_main_process, init_distributed_mode
from Model.il_trainer import VLNCETrainer
from Model.utils.tensor_dict import DictTree, TensorDict
from Model.aux_losses import AuxLosses
from Model.utils.tensorboard_utils import TensorboardWriter
from Model.utils.common import observations_to_image, append_text_to_image, generate_video

from src.common.param import args
from src.vlnce_src.env import AirVLNENV
from src.vlnce_src.util import read_vocab, Tokenizer


def setup():
    init_distributed_mode()

    seed = 100 + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False


class DDPIWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=5.0e12,
        batch_size=1,
    ):
        super().__init__()

        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        self.keys = []
        self.seed = 1

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
            readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                self.keys.append(key.decode())

        self.length = len(self.keys)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.start = 0
        self.end = self.length

        self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        self.iter_start = 0 + self.rank * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)
        logger.warning("END init DDP-Dataset \t rank: {} \t start({}) - end({})".format(self.rank, self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i+1) % 10 == 0:
                        logger.warning("rank: {} \t lmdb load: {} / {}".format(self.rank, i+1, self.preload_size))

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.keys[self.load_ordering.pop()]).encode()),
                            raw=False,
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(self.iter_start, self.iter_end)), self.preload_size)
            )
        )

        return self


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=5.0e12,
        batch_size=1,
    ):
        super().__init__()

        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        self.keys = []
        self.seed = 1

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
            readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                self.keys.append(key.decode())

        self.length = len(self.keys)

        self.iter_start = 0
        self.iter_end = self.length
        logger.warning("END init Dataset \t start({}) - end({})".format(self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i+1) % 10 == 0:
                        if self.worker_info is not None:
                            logger.info("{} lmdb load: {} / {}".format(self.worker_info.id, i+1, self.preload_size))
                        else:
                            logger.info("{} lmdb load: {} / {}".format(0, i+1, self.preload_size))

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.keys[self.load_ordering.pop()]).encode()),
                            raw=False,
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.worker_info = worker_info
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    # max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    max_traj_len = 500
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid][:max_traj_len, ...], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid][:max_traj_len, ...], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


@torch.no_grad()
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))


def initialize_tokenizer():
    if args.tokenizer_use_bert:
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        vocab = read_vocab(args.TRAIN_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    return tok


def initialize_env(split='train'):
    tok = initialize_tokenizer()

    train_env = AirVLNENV(batch_size=args.batchSize, split=split, tokenizer=tok)

    return train_env


def initialize_trainer():
    from gym import spaces
    from airsim_plugin.airsim_settings import AirsimActions

    observation_space = spaces.Dict({
        "rgb": spaces.Box(low=0, high=255, shape=(args.Image_Height_RGB, args.Image_Width_RGB, 3), dtype=np.uint8),
        "depth": spaces.Box(low=0, high=1, shape=(args.Image_Height_DEPTH, args.Image_Width_DEPTH, 1), dtype=np.float32),
        "instruction": spaces.Discrete(0),
        "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
    })
    action_space = spaces.Discrete(int(len(AirsimActions)))

    trainer = VLNCETrainer(
        load_from_ckpt=False,
        observation_space=observation_space,
        action_space=action_space,
    )

    logger.info('initialize_trainer over')
    return trainer


def collect_data(data_it=0):
    logger.info(args)

    train_env = initialize_env(split='train')
    trainer = initialize_trainer()

    if torch.cuda.is_available():
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()

    def hook_builder(tgt_tensor):
        def hook(m, i, o):
            tgt_tensor.set_(o.cpu())

        return hook

    rgb_features = torch.zeros((1,), device="cpu")
    if not args.ablate_rgb:
        rgb_hook = trainer.policy.net.rgb_encoder.layer_extract.register_forward_hook(
            hook_builder(rgb_features)
        )
    else:
        rgb_hook = None

    depth_features = torch.zeros((1,), device="cpu")
    if not args.ablate_depth:
        depth_hook = trainer.policy.net.depth_encoder.visual_encoder.register_forward_hook(
            hook_builder(depth_features)
        )
    else:
        depth_hook = None

    p = 1.0
    beta = 1.0


    #
    with torch.no_grad():
        end_iter = len(train_env.data)
        pbar = None
        pbar_pre_index = 0
        while train_env.index_data < end_iter:
            if pbar_pre_index + train_env.batch_size >= end_iter:
                break

            pbar_pre_index = train_env.index_data
            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break

            if pbar is None:
                pbar = tqdm.tqdm(total=end_iter)
                pbar.update(train_env.index_data)
            else:
                pbar.update(n=train_env.index_data-pbar_pre_index)

            if args.policy_type in ['seq2seq', 'cma', 'unet', 'vlnbert']:
                rnn_states = torch.zeros(
                    train_env.batch_size,
                    trainer.policy.net.num_recurrent_layers,
                    trainer.policy.net.state_encoder.hidden_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            elif args.policy_type in ['hcm']:
                rnn_states = torch.zeros(
                    trainer.policy.net.num_recurrent_layers,
                    train_env.batch_size,
                    trainer.policy.net.state_encoder.hidden_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            else:
                raise NotImplementedError

            episodes = [[] for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            envs_to_pause = []

            outputs = train_env.reset()
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, trainer.device)

            ended = False

            for t in range(int(args.maxAction) + 1):
                logger.info('{} - {} / {}'.format(int(train_env.index_data)-int(train_env.batch_size), t, end_iter))

                for i in range(train_env.batch_size):
                    if dones[i] and not skips[i]:
                        if args.collect_type in ['TF']:
                            _episodes = episodes[i].copy()
                            for _i, _j in enumerate(train_env.trajectory_id_2_instruction_tokens[infos[i]['trajectory_id']]):
                                for __i, __j in enumerate(_episodes):
                                    _episodes[__i][0]['instruction'] = _j

                                ep = _episodes.copy()
                                traj_obs = batch_obs(
                                    [step[0] for step in ep],
                                    device=torch.device("cpu"),
                                )
                                del traj_obs['teacher_action']
                                for k, v in traj_obs.items():
                                    traj_obs[k] = v.numpy()

                                transposed_ep = [
                                    traj_obs,
                                    np.array([step[1] for step in ep], dtype=np.int64),
                                    np.array([step[2] for step in ep], dtype=np.int64),
                                ]

                                train_env.threading_lock_lmdb_features_txn.acquire()
                                lmdb_key = str(train_env.trajectory_id_2_episode_ids[infos[i]['trajectory_id']][_i])
                                train_env.lmdb_features_txn.put(
                                    lmdb_key.encode(),
                                    msgpack_numpy.packb(
                                        transposed_ep, use_bin_type=True
                                    ),
                                )
                                train_env.lmdb_features_txn.commit()
                                train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                                train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                                train_env.threading_lock_lmdb_features_txn.release()
                                logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                            if args.run_type in ['collect'] and args.collect_type in ['TF']:
                                train_env.threading_lock_lmdb_rgb_txn.acquire()
                                train_env.lmdb_rgb_txn.commit()
                                train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                                train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                                train_env.threading_lock_lmdb_rgb_txn.release()

                                train_env.threading_lock_lmdb_depth_txn.acquire()
                                train_env.lmdb_depth_txn.commit()
                                train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                                train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                                train_env.threading_lock_lmdb_depth_txn.release()

                            episodes[i] = []
                            _episodes = []
                            envs_to_pause.append(i)
                            skips[i] = True

                        else:
                            ep = episodes[i]
                            traj_obs = batch_obs(
                                [step[0] for step in ep],
                                device=torch.device("cpu"),
                            )
                            del traj_obs['teacher_action']
                            for k, v in traj_obs.items():
                                traj_obs[k] = v.numpy()

                            transposed_ep = [
                                traj_obs,
                                np.array([step[1] for step in ep], dtype=np.int64),
                                np.array([step[2] for step in ep], dtype=np.int64),
                            ]

                            train_env.threading_lock_lmdb_features_txn.acquire()
                            lmdb_key = str(infos[i]['episode_id'])
                            train_env.lmdb_features_txn.put(
                                lmdb_key.encode(),
                                msgpack_numpy.packb(
                                    transposed_ep, use_bin_type=True
                                ),
                            )
                            train_env.lmdb_features_txn.commit()
                            train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                            train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                            train_env.lmdb_collected_keys.add(lmdb_key)
                            train_env.threading_lock_lmdb_features_txn.release()
                            logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                            if args.run_type in ['collect'] and args.collect_type in ['TF']:
                                train_env.threading_lock_lmdb_rgb_txn.acquire()
                                train_env.lmdb_rgb_txn.commit()
                                train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                                train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                                train_env.threading_lock_lmdb_rgb_txn.release()

                                train_env.threading_lock_lmdb_depth_txn.acquire()
                                train_env.lmdb_depth_txn.commit()
                                train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                                train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                                train_env.threading_lock_lmdb_depth_txn.release()

                            episodes[i] = []
                            envs_to_pause.append(i)
                            skips[i] = True

                    if np.array(dones).all():
                        ended = True

                if ended:
                    break

                actions, rnn_states = trainer.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    batch['teacher_action'].long(),
                    actions,
                )

                for i in range(train_env.batch_size):
                    if not args.ablate_rgb and rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]
                        del observations[i]["rgb"]

                    if not args.ablate_depth and depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]

                    if i in envs_to_pause:
                        continue

                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch['teacher_action'][i].item(),
                        )
                    )

                prev_actions.copy_(actions)

                # Make action and get the new state
                actions = [temp[0] for temp in actions.cpu().numpy()]
                train_env.makeActions(actions)

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, trainer.device)

                logger.info('action: {}'.format(actions))

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=trainer.device,
                )

            for i in range(train_env.batch_size):
                if dones[i] and not t >= int(args.maxAction):
                    continue

                if args.collect_type in ['TF']:
                    _episodes = episodes[i].copy()
                    for _i, _j in enumerate(train_env.trajectory_id_2_instruction_tokens[infos[i]['trajectory_id']]):
                        for __i, __j in enumerate(_episodes):
                            _episodes[__i][0]['instruction'] = _j

                        ep = _episodes.copy()
                        if len(ep) <= 0:
                            continue
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        )
                        del traj_obs['teacher_action']
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]

                        train_env.threading_lock_lmdb_features_txn.acquire()
                        lmdb_key = str(train_env.trajectory_id_2_episode_ids[infos[i]['trajectory_id']][_i])
                        train_env.lmdb_features_txn.put(
                            lmdb_key.encode(),
                            msgpack_numpy.packb(
                                transposed_ep, use_bin_type=True
                            ),
                        )
                        train_env.lmdb_features_txn.commit()
                        train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                        train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                        train_env.lmdb_collected_keys.add(lmdb_key)
                        train_env.threading_lock_lmdb_features_txn.release()
                        logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                    if args.run_type in ['collect'] and args.collect_type in ['TF']:
                        train_env.threading_lock_lmdb_rgb_txn.acquire()
                        train_env.lmdb_rgb_txn.commit()
                        train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                        train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                        train_env.threading_lock_lmdb_rgb_txn.release()

                        train_env.threading_lock_lmdb_depth_txn.acquire()
                        train_env.lmdb_depth_txn.commit()
                        train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                        train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                        train_env.threading_lock_lmdb_depth_txn.release()

                    episodes[i] = []
                    _episodes = []
                    envs_to_pause.append(i)
                    skips[i] = True

                else:
                    ep = episodes[i]
                    if len(ep) <= 0:
                        continue
                    traj_obs = batch_obs(
                        [step[0] for step in ep],
                        device=torch.device("cpu"),
                    )
                    del traj_obs['teacher_action']
                    for k, v in traj_obs.items():
                        traj_obs[k] = v.numpy()

                    transposed_ep = [
                        traj_obs,
                        np.array([step[1] for step in ep], dtype=np.int64),
                        np.array([step[2] for step in ep], dtype=np.int64),
                    ]

                    train_env.threading_lock_lmdb_features_txn.acquire()
                    lmdb_key = str(infos[i]['episode_id'])
                    train_env.lmdb_features_txn.put(
                        lmdb_key.encode(),
                        msgpack_numpy.packb(
                            transposed_ep, use_bin_type=True
                        ),
                    )
                    train_env.lmdb_features_txn.commit()
                    train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                    train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                    train_env.lmdb_collected_keys.add(lmdb_key)
                    train_env.threading_lock_lmdb_features_txn.release()
                    logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                    if args.run_type in ['collect'] and args.collect_type in ['TF']:
                        train_env.threading_lock_lmdb_rgb_txn.acquire()
                        train_env.lmdb_rgb_txn.commit()
                        train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                        train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                        train_env.threading_lock_lmdb_rgb_txn.release()

                        train_env.threading_lock_lmdb_depth_txn.acquire()
                        train_env.lmdb_depth_txn.commit()
                        train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                        train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                        train_env.threading_lock_lmdb_depth_txn.release()

                    episodes[i] = []
                    envs_to_pause.append(i)
                    skips[i] = True


    if rgb_hook is not None:
        rgb_hook.remove()
    if depth_hook is not None:
        depth_hook.remove()

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass
    logger.info('END data_it: {}'.format(data_it))


def train_vlnce():
    logger.info(args)

    if get_rank() == 0:
        writer = SummaryWriter(
            log_dir=str(Path(args.project_prefix) / 'DATA/output/{}/train/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        )
    else:
        writer = None

    trainer = initialize_trainer()

    for dagger_it in range(int(args.dagger_it)):
        step_id = 0

        if torch.cuda.is_available():
            with torch.cuda.device(trainer.device):
                torch.cuda.empty_cache()
        gc.collect()

        lmdb_features_dir = str(Path(args.project_prefix) / 'DATA/img_features/collect/{}/train'.format(args.name))
        assert os.path.exists(str(lmdb_features_dir))
        if args.DistributedDataParallel:
            dataset = DDPIWTrajectoryDataset(
                lmdb_features_dir,
                use_iw=True,
                inflection_weight_coef=float(args.inflection_weight_coef),
                lmdb_map_size=5.0e12,
                batch_size=args.batchSize,
            )
            diter = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batchSize,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=True,
                num_workers=0,
            )
        else:
            dataset = IWTrajectoryDataset(
                lmdb_features_dir,
                use_iw=True,
                inflection_weight_coef=float(args.inflection_weight_coef),
                lmdb_map_size=5.0e12,
                batch_size=args.batchSize,
            )
            diter = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batchSize,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=True,
                num_workers=0,
            )

        AuxLosses.activate()
        for epoch in tqdm.trange(int(args.epochs), dynamic_ncols=True):
            batch_cnt = 0
            for batch in tqdm.tqdm(
                diter,
                total=dataset.length // dataset.batch_size if not args.DistributedDataParallel else (dataset.iter_end - dataset.iter_start) // dataset.batch_size,
                leave=False,
                dynamic_ncols=True,
            ):
                (
                    observations_batch,
                    prev_actions_batch,
                    not_done_masks,
                    corrected_actions_batch,
                    weights_batch,
                ) = batch

                observations_batch = {
                    k: v.to(
                        device=trainer.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    for k, v in observations_batch.items()
                }

                if args.policy_type in ['vlnbert']:
                    loss, action_loss, aux_loss = trainer._update_agent(
                        observations_batch,
                        prev_actions_batch.view(-1, args.batchSize)[:args.maxAction, :].view(-1, 1).to(
                            device=trainer.device, non_blocking=True
                        ),
                        not_done_masks.view(-1, args.batchSize)[:args.maxAction, :].view(-1, 1).to(
                            device=trainer.device, non_blocking=True
                        ),
                        corrected_actions_batch.view(-1, args.batchSize)[:args.maxAction, :].view(-1, 1).to(
                            device=trainer.device, non_blocking=True
                        ),
                        weights_batch.view(-1, args.batchSize)[:args.maxAction, :].view(-1, 1).to(
                            device=trainer.device, non_blocking=True
                        ),
                    )
                else:
                    loss, action_loss, aux_loss = trainer._update_agent(
                        observations_batch,
                        prev_actions_batch.to(
                            device=trainer.device, non_blocking=True
                        ),
                        not_done_masks.to(
                            device=trainer.device, non_blocking=True
                        ),
                        corrected_actions_batch.to(
                            device=trainer.device, non_blocking=True
                        ),
                        weights_batch.to(
                            device=trainer.device, non_blocking=True
                        ),
                    )

                logger.warning(
                    'dagger_it: {} / {} \t epoch: {} / {} \t batch: {} / {}'.format(
                        dagger_it, args.dagger_it,
                        epoch, args.epochs,
                        batch_cnt, dataset.length // dataset.batch_size
                    )
                )

                logger.info(f"train_loss: {loss}")
                logger.info(f"train_action_loss: {action_loss}")
                logger.info(f"train_aux_loss: {aux_loss}")
                logger.info(f"Batches processed: {step_id}.")
                logger.info(
                    f"On DAgger iter {dagger_it}, Epoch {epoch}."
                )
                logger.info('\n')

                if get_rank() == 0:
                    writer.add_scalar(
                        f"train_loss_iter_{dagger_it}", loss, step_id
                    )
                    writer.add_scalar(
                        f"train_action_loss_iter_{dagger_it}",
                        action_loss,
                        step_id,
                    )
                    writer.add_scalar(
                        f"train_aux_loss_iter_{dagger_it}",
                        aux_loss,
                        step_id,
                    )

                step_id += 1
                batch_cnt += 1

            if is_main_process():
                if args.policy_type in ['vlnbert'] or \
                        ((dagger_it * args.epochs + epoch)+1) % 5 == 0:
                    trainer.save_checkpoint(
                        f"ckpt.{dagger_it * args.epochs + epoch}.pth",
                        dagger_it,
                        epoch,
                    )

            if is_dist_avail_and_initialized() == 1:
                dist.barrier()

        if is_main_process():
            trainer.save_checkpoint(
                f"ckpt.LAST.pth",
                dagger_it,
                epoch,
            )
        AuxLosses.deactivate()


def eval_vlnce():
    logger.info(args)

    writer = TensorboardWriter(
        str(Path(args.project_prefix) / 'DATA/output/{}/eval/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        flush_secs=30,
    )

    tok = initialize_tokenizer()

    assert os.path.exists(args.EVAL_CKPT_PATH_DIR), '评估文件(夹)不存在'
    if os.path.isfile(args.EVAL_CKPT_PATH_DIR):
        from Model.utils.common import get_checkpoint_id

        # evaluate singe checkpoint
        proposed_index = get_checkpoint_id(args.EVAL_CKPT_PATH_DIR)
        if proposed_index is not None:
            ckpt_idx = proposed_index
        else:
            ckpt_idx = 100000

        _eval_checkpoint(
            checkpoint_path=args.EVAL_CKPT_PATH_DIR,
            writer=writer,
            tok=tok,
            checkpoint_index=ckpt_idx,
        )
        logger.info("END evaluate")
    else:
        from Model.utils.common import poll_checkpoint_folder

        # evaluate multiple checkpoints in order
        prev_ckpt_ind = -1
        while True:
            current_ckpt = None
            while current_ckpt is None:
                current_ckpt = poll_checkpoint_folder(
                    args.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                )
                time.sleep(2)
            logger.info(f"=======current_ckpt: {current_ckpt}=======")
            prev_ckpt_ind += 1

            # 跳过
            if prev_ckpt_ind <= 2:
                continue

            _eval_checkpoint(
                checkpoint_path=current_ckpt,
                writer=writer,
                tok=tok,
                checkpoint_index=prev_ckpt_ind,
            )

    if writer is not None:
        try:
            writer.writer.close()
            del writer
        except Exception as e:
            logger.error(e)
    logger.info("END evaluate")


def _eval_checkpoint(
    checkpoint_path: str,
    writer,
    tok,
    checkpoint_index: int = 0,
) -> None:
    logger.info(f"checkpoint_path: {checkpoint_path}")


    if args.EVAL_DATASET == 'train':
        train_env = AirVLNENV(batch_size=args.batchSize, split='train', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_seen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_seen', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_unseen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_unseen', tokenizer=tok)
    elif args.EVAL_DATASET == 'test':
        train_env = AirVLNENV(batch_size=args.batchSize, split='test', tokenizer=tok)
    else:
        raise KeyError


    #
    EVAL_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/results/{}'.format(args.name, args.make_dir_time)
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if os.path.exists(fname):
        print("skipping -- evaluation exists.")
        return


    #
    trainer = VLNCETrainer(
        load_from_ckpt=True,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        ckpt_path=checkpoint_path,
    )
    trainer.policy.eval()

    if torch.cuda.is_available():
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()
    gc.collect()


    #
    stats_episodes = {}
    episodes_to_eval = len(train_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    with torch.no_grad():
        start_iter = 0
        end_iter = len(train_env.data)
        cnt = 0
        if args.TF_test_one_scene:
            flag_error_cnt = 0
            flag_error_total_cnt = 0
            TF_test_one_scene_progress_list = []
        for idx in range(start_iter, end_iter, train_env.batch_size):
            if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
                break
            cnt += 1

            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break

            if args.policy_type in ['seq2seq', 'cma']:
                rnn_states = torch.zeros(
                    train_env.batch_size,
                    trainer.policy.net.num_recurrent_layers,
                    trainer.policy.net.state_encoder.hidden_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            elif args.policy_type in ['hcm']:
                rnn_states = torch.zeros(
                    trainer.policy.net.num_recurrent_layers,
                    train_env.batch_size,
                    trainer.policy.net.state_encoder.hidden_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            elif args.policy_type in ['unet']:
                rnn_states = torch.zeros(
                    train_env.batch_size,
                    trainer.policy.net.num_recurrent_layers,
                    trainer.policy.net.output_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            elif args.policy_type in ['vlnbert']:
                raise NotImplementedError
            else:
                raise NotImplementedError

            rgb_frames = [[] for _ in range(train_env.batch_size)]

            episodes = [[] for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            envs_to_pause = []
            if args.TF_test_one_scene:
                flag_error = [False for _ in range(train_env.batch_size)]

            outputs = train_env.reset()
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, trainer.device)

            ended = False

            for t in range(int(args.maxAction)):
                logger.info('checkpoint_index:{} \t {} - {} / {} \t {}'.format(checkpoint_index, idx, t, end_iter, not_done_masks.cpu().numpy().reshape((-1,)).tolist()))

                actions, rnn_states = trainer.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                    step=t,
                )
                prev_actions.copy_(actions)

                # Make action and get the new state
                actions = [temp[0] for temp in actions.cpu().numpy()]
                if args.TF_test_one_scene:
                    for action_index, action in enumerate(actions):
                        if action != train_env.batch[action_index]['actions'][train_env.sim_states[action_index].step]:
                            flag_error[action_index] = True
                            train_env.sim_states[action_index].is_end = True
                train_env.makeActions(actions)

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, trainer.device)

                logger.info('action: {}'.format(actions))

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=trainer.device,
                )

                # for tttt in range(len(train_env.batch)):
                #     train_env.threading_lock_lmdb_features_txn.acquire()
                #     train_env.lmdb_features_txn.put(
                #         str('{}_{}_{}'.format(infos[tttt]['episode_id'], t, 'exp_1')).encode(),
                #         msgpack_numpy.packb(
                #             observations[tttt], use_bin_type=True
                #         ),
                #     )
                #     train_env.lmdb_features_txn.commit()
                #     train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                #     train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                #     train_env.threading_lock_lmdb_features_txn.release()
                #     logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                # reset envs and observations if necessary
                for i in range(train_env.batch_size):
                    if args.EVAL_GENERATE_VIDEO:
                        frame = observations_to_image(observations[i], infos[i])
                        frame = append_text_to_image(
                            frame, train_env.batch[i]['instruction']['instruction_text']
                        )
                        rgb_frames[i].append(frame)

                    if not dones[i] or skips[i]:
                        continue

                    skips[i] = True
                    pbar.update()

                if np.array(dones).all():
                    ended = True
                    break

            for t in range(int(train_env.batch_size)):
                stats_episodes[str(train_env.batch[t]['episode_id'])] = infos[t]

                EVAL_SAVE_EVERY_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results_every/{}'.format(args.name, args.make_dir_time)
                if not os.path.exists(str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index))):
                    os.makedirs(str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index)), exist_ok=True)

                f_intermediate_result_name = os.path.join(
                    str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index)),
                    f"{train_env.batch[t]['episode_id']}.json",
                )
                f_intermediate_trajectory = {**infos[t]}
                with open(f_intermediate_result_name, "w") as f:
                    json.dump(f_intermediate_trajectory, f)

                if args.EVAL_GENERATE_VIDEO:
                    EVAL_GENERATE_VIDEO_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/videos/{}'.format(args.name, args.make_dir_time)
                    generate_video(
                        video_option=["disk"],
                        video_dir=str(EVAL_GENERATE_VIDEO_DIR),
                        images=rgb_frames[t],
                        episode_id=train_env.batch[t]['episode_id'],
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            # "spl": infos[t]['spl'],
                            "ndtw": infos[t]['ndtw'],
                        },
                        tb_writer=writer,
                    )

                logger.info((
                    'result-{} \t' +
                    'distance_to_goal: {} \t' +
                    'oracle_navigation_error: {} \t' +
                    'success: {} \t' +
                    'spl: {} \t' +
                    'soft_spl: {} \t' +
                    'oracle_spl: {} \t' +
                    'ndtw: {} \t' +
                    'sdtw: {} \t' +
                    'path_length: {} \t' +
                    'oracle_success: {} \t' +
                    'steps_taken: {}'
                ).format(
                    t,
                    infos[t]['distance_to_goal'],
                    infos[t]['oracle_navigation_error'],
                    infos[t]['success'],
                    infos[t]['spl'],
                    infos[t]['soft_spl'],
                    infos[t]['oracle_spl'],
                    infos[t]['ndtw'],
                    infos[t]['sdtw'],
                    infos[t]['path_length'],
                    infos[t]['oracle_success'],
                    infos[t]['steps_taken']
                ))


            if args.TF_test_one_scene:
                flag_error_cnt += int(np.array(flag_error).sum())
                flag_error_total_cnt += len(flag_error)
                logger.info("flag_error_cnt: {} \t flag_error_total_cnt: {}".format(flag_error_cnt, flag_error_total_cnt))

                for t in range(int(train_env.batch_size)):
                    logger.info('process-{}/{}'.format(
                        train_env.sim_states[t].step,
                        len(train_env.batch[t]['actions']),
                    ))
                    TF_test_one_scene_progress_list.append([
                        train_env.batch[t]['episode_id'], train_env.sim_states[t].step, len(train_env.batch[t]['actions'])
                    ])

    # end
    pbar.close()


    #
    EVAL_INTERMEDIATE_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results/{}'.format(args.name, args.make_dir_time)
    f_intermediate_name = os.path.join(
        EVAL_INTERMEDIATE_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_INTERMEDIATE_RESULTS_DIR):
        os.makedirs(EVAL_INTERMEDIATE_RESULTS_DIR, exist_ok=True)
    with open(f_intermediate_name, "w") as f:
        json.dump(stats_episodes, f)

    #
    new_stats_episodes = {}
    for i, j in stats_episodes.items():
        temp_1 = {}
        temp_1 = j.copy()

        temp_2 = temp_1.copy()
        for _i, _j in temp_2.items():
            if type(_j) == str or type(_j) == list or type(_j) == dict:
                del temp_1[_i]

        new_stats_episodes[i] = temp_1.copy()
    stats_episodes = new_stats_episodes.copy()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = (
            sum(v[stat_key] for v in stats_episodes.values())
            / num_episodes
        )

    #
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(aggregated_stats, f, indent=4)

    logger.info(f"Episodes evaluated: {num_episodes}")
    checkpoint_num = checkpoint_index + 1
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.6f}")
        writer.add_scalar(f"eval_{train_env.split}_{k}", v, checkpoint_num)

    if args.TF_test_one_scene:
        logger.info("flag_error_cnt: {} \t flag_error_total_cnt: {}".format(flag_error_cnt, flag_error_total_cnt))

        p_progress_1 = (np.array(TF_test_one_scene_progress_list)[:, 1]).astype(float).sum() / (np.array(TF_test_one_scene_progress_list)[:, 2]).astype(float).sum()
        logger.info("average action progress_1: {}".format(p_progress_1))

        p_progress_2 = ((np.array(TF_test_one_scene_progress_list)[:, 1]).astype(float) / (np.array(TF_test_one_scene_progress_list)[:, 2]).astype(float)).sum() / len(TF_test_one_scene_progress_list)
        logger.info("average action progress_2: {}".format(p_progress_2))

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass


def eval_random_agent():
    logger.info(args)

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

    writer = SummaryWriter(
        log_dir=str(
            Path(args.project_prefix) / 'DATA/output/{}/eval/TensorBoard/{}'.format(args.name, args.make_dir_time)),
    )

    tok = initialize_tokenizer()

    if args.EVAL_DATASET == 'train':
        train_env = AirVLNENV(batch_size=args.batchSize, split='train', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_seen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_seen', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_unseen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_unseen', tokenizer=tok)
    elif args.EVAL_DATASET == 'test':
        train_env = AirVLNENV(batch_size=args.batchSize, split='test', tokenizer=tok)
    else:
        raise KeyError


    #
    EVAL_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/results/{}'.format(args.name, args.make_dir_time)
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{0}_{train_env.split}.json",
    )
    if os.path.exists(fname):
        print("skipping -- evaluation exists.")
        return


    #
    device = (
        torch.device("cuda", args.trainer_gpu_device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    gc.collect()


    # eval
    stats_episodes = {}
    episodes_to_eval = len(train_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    with torch.no_grad():
        start_iter = 0
        end_iter = len(train_env.data)
        cnt = 0
        for idx in range(start_iter, end_iter, train_env.batch_size):
            if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
                break
            cnt += 1

            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break

            not_done_masks = torch.zeros(
                train_env.batch_size,
                1,
                dtype=torch.uint8,
                device=device,
            )

            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]

            outputs = train_env.reset()
            _, _, dones, infos = [list(x) for x in zip(*outputs)]

            for t in range(int(args.maxAction) + 1):
                logger.info('checkpoint_index:{} \t {} - {} / {} \t {}'.format(0, idx, t, end_iter, not_done_masks.cpu().numpy().reshape((-1,)).tolist()))

                action_probs = [(action_prob / np.array(action_probs).sum()) for action_prob in action_probs]
                all_actions = [0, 1, 2, 3, 4, 5, 6, 7]
                actions = [np.random.choice(all_actions, p=action_probs) for _ in range(train_env.batch_size)]

                # Make action and get the new state
                train_env.makeActions(actions)

                outputs = train_env.get_obs()
                _, _, dones, infos = [list(x) for x in zip(*outputs)]

                logger.info('action: {}'.format(actions))

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=device,
                )

                # reset envs and observations if necessary
                for i in range(train_env.batch_size):
                    if not dones[i] or skips[i]:
                        continue

                    skips[i] = True

                if np.array(dones).all():
                    break

            for t in range(int(train_env.batch_size)):
                stats_episodes[str(train_env.batch[t]['episode_id'])] = infos[t]

                #
                EVAL_SAVE_EVERY_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results_every/{}'.format(args.name, args.make_dir_time)
                if not os.path.exists(str(Path(str(EVAL_SAVE_EVERY_RESULTS_DIR)) / str(0))):
                    os.makedirs(str(Path(str(EVAL_SAVE_EVERY_RESULTS_DIR)) / str(0)), exist_ok=True)

                f_intermediate_result_name = os.path.join(
                    str(EVAL_SAVE_EVERY_RESULTS_DIR / str(0)),
                    f"{train_env.batch[t]['episode_id']}.json",
                )
                f_intermediate_trajectory = {**infos[t]}
                with open(f_intermediate_result_name, "w") as f:
                    json.dump(f_intermediate_trajectory, f)

                logger.info(
                    ('result-{} \t' +
                     'distance_to_goal: {} \t' +
                     'oracle_navigation_error: {} \t' +
                     'success: {} \t' +
                     'spl: {} \t' +
                     'soft_spl: {} \t' +
                     'oracle_spl: {} \t' +
                     'ndtw: {} \t' +
                     'sdtw: {} \t' +
                     'path_length: {} \t' +
                     'oracle_success: {} \t' +
                     'steps_taken: {}'
                     ).format(
                        t,
                        infos[t]['distance_to_goal'],
                        infos[t]['oracle_navigation_error'],
                        infos[t]['success'],
                        infos[t]['spl'],
                        infos[t]['soft_spl'],
                        infos[t]['oracle_spl'],
                        infos[t]['ndtw'],
                        infos[t]['sdtw'],
                        infos[t]['path_length'],
                        infos[t]['oracle_success'],
                        infos[t]['steps_taken']
                    )
                )


    # end
    pbar.close()

    #
    EVAL_INTERMEDIATE_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results/{}'.format(args.name, args.make_dir_time)
    f_intermediate_name = os.path.join(
        EVAL_INTERMEDIATE_RESULTS_DIR,
        f"stats_ckpt_{0}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_INTERMEDIATE_RESULTS_DIR):
        os.makedirs(EVAL_INTERMEDIATE_RESULTS_DIR, exist_ok=True)
    with open(f_intermediate_name, "w") as f:
        json.dump(stats_episodes, f)

    #
    new_stats_episodes = {}
    for i, j in stats_episodes.items():
        temp_1 = {}
        temp_1 = j.copy()

        temp_2 = temp_1.copy()
        for _i, _j in temp_2.items():
            if type(_j) == str or type(_j) == list or type(_j) == dict:
                del temp_1[_i]

        new_stats_episodes[i] = temp_1.copy()
    stats_episodes = new_stats_episodes.copy()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = (
            sum(v[stat_key] for v in stats_episodes.values())
            / num_episodes
        )

    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{0}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(aggregated_stats, f, indent=4)

    logger.info(f"Episodes evaluated: {num_episodes}")
    checkpoint_num = 1
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.6f}")
        writer.add_scalar(f"eval_random_agent_{k}", v, checkpoint_num)

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass


if __name__ == "__main__":
    setup()

    if args.run_type == 'collect':
        collect_data()
    elif args.run_type == 'train':
        train_vlnce()
    elif args.run_type == 'eval':
        eval_vlnce()
    elif args.run_type == 'eval_random_agent':
        eval_random_agent()
    else:
        raise NotImplementedError
