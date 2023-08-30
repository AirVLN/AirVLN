import argparse
import os
import datetime
from pathlib import Path
from utils.CN import CN


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        project_prefix = Path(str(os.getcwd())).parent.resolve()
        self.parser.add_argument('--project_prefix', type=str, default=str(project_prefix), help="project path")

        self.parser.add_argument('--run_type', type=str, default="train", help="run_type in [collect, train, eval]")
        self.parser.add_argument('--policy_type', type=str, default="seq2seq", help="policy_type in [seq2seq, cma, hcm, unet, vlnbert]")
        self.parser.add_argument('--collect_type', type=str, default="TF", help="seq2seq in [TF, dagger, SF]")
        self.parser.add_argument('--name', type=str, default='default', help='experiment name')

        self.parser.add_argument('--maxInput', type=int, default=300, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=500, help='max action sequence')

        self.parser.add_argument("--dagger_it", type=int, default=1)
        self.parser.add_argument("--epochs", type=int, default=10)
        self.parser.add_argument('--lr', type=float, default=0.00025, help="learning rate")
        self.parser.add_argument('--batchSize', type=int, default=8)
        self.parser.add_argument("--trainer_gpu_device", type=int, default=0, help='GPU')

        self.parser.add_argument('--Image_Height_RGB', type=int, default=224)
        self.parser.add_argument('--Image_Width_RGB', type=int, default=224)
        self.parser.add_argument('--Image_Height_DEPTH', type=int, default=256)
        self.parser.add_argument('--Image_Width_DEPTH', type=int, default=256)

        self.parser.add_argument('--inflection_weight_coef', type=float, default=1.9)

        self.parser.add_argument('--nav_graph_path', type=str, default=str(project_prefix / 'DATA/data/disceret/processed/nav_graph_10'), help="nav_graph path")
        self.parser.add_argument('--token_dict_path', type=str, default=str(project_prefix / 'DATA/data/disceret/processed/token_dict_10'), help="token_dict path")
        self.parser.add_argument('--vertices_path', type=str, default=str(project_prefix / 'DATA/data/disceret/scene_meshes'))
        self.parser.add_argument('--dagger_mode_load_scene', nargs='+', default=[])
        self.parser.add_argument('--dagger_update_size', type=int, default=8000)
        self.parser.add_argument('--dagger_mode', type=str, default="end", help='dagger mode in [end middle nearest]')
        self.parser.add_argument('--dagger_p', type=float, default=1.0, help='dagger p')

        self.parser.add_argument('--TF_mode_load_scene', nargs='+', default=[])
        self.parser.add_argument('--TF_test_one_scene', action="store_true")

        self.parser.add_argument('--SUCCESS_DISTANCE_SCALE', type=float, default=0.3)

        self.parser.add_argument('--ablate_instruction', action="store_true")
        self.parser.add_argument('--ablate_rgb', action="store_true")
        self.parser.add_argument('--ablate_depth', action="store_true")
        self.parser.add_argument('--SEQ2SEQ_use_prev_action', action="store_true")
        self.parser.add_argument('--PROGRESS_MONITOR_use', action="store_true")
        self.parser.add_argument('--PROGRESS_MONITOR_alpha', type=float, default=1.0)

        self.parser.add_argument('--EVAL_CKPT_PATH_DIR', type=str)
        self.parser.add_argument('--EVAL_DATASET', type=str, default="val_unseen")
        self.parser.add_argument("--EVAL_NUM", type=int, default=-1)
        self.parser.add_argument('--EVAL_GENERATE_VIDEO', action="store_true")

        self.parser.add_argument('--rgb_encoder_use_place365', action="store_true")
        self.parser.add_argument('--tokenizer_use_bert', action="store_true")

        self.parser.add_argument("--simulator_tool_port", type=int, default=30000, help="simulator_tool port")
        self.parser.add_argument("--DDP_MASTER_PORT", type=int, default=20000, help="DDP MASTER_PORT")

        self.parser.add_argument('--collision_sensor_disabled', action="store_false")

        self.parser.add_argument("--continue_start_from_dagger_it", type=int)
        self.parser.add_argument("--continue_start_from_checkpoint_path", type=str)

        self.parser.add_argument('--vlnbert', action="store_true", default="prevalent")
        self.parser.add_argument('--featdropout', action="store_true", default=0.4)
        self.parser.add_argument('--action_feature', action="store_true", default=32)

        self.args = self.parser.parse_args()


param = Param()
args = param.args

args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
args.logger_file_name = '{}/DATA/output/{}/{}/logs/{}_{}.log'.format(args.project_prefix, args.name, args.run_type, args.name, args.make_dir_time)


# args.run_type = 'collect'
assert args.run_type in ['collect', 'train', 'eval', 'seg_lmdb', 'eval_random_agent'], 'run_type error'
# args.policy_type = 'seq2seq'
assert args.policy_type in ['seq2seq', 'cma', 'hcm', 'unet', 'vlnbert'], 'policy_type error'
# args.collect_type = 'TF'
assert args.collect_type in ['TF', 'dagger'], 'collect_type error'


args.machines_info = [
    {
        'MACHINE_IP': '127.0.0.1',
        'SOCKET_PORT': int(args.simulator_tool_port),
        'MAX_SCENE_NUM': 16,
        'open_scenes': [],
    },
]


args.TRAIN_VOCAB = Path(args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.TRAINVAL_VOCAB = Path(args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.vocab_size = 10038


default_config = CN.clone()
default_config.make_dir_time = args.make_dir_time
default_config.freeze()

