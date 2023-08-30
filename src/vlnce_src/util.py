import sys
import re
import string
import json
import time
import math
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
from numpy.linalg import norm

from src.common.param import args
from utils.logger import logger


# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>']
padding_idx = base_vocab.index('<PAD>')


def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_datasets(splits):
    """

    :param splits: A list of split.
        if the split is "something@5000", it will use a random 5000 data from the data
    :return:
    """
    import random
    data = []
    old_state = random.getstate()
    for split in splits:
        # It only needs some part of the dataset?
        components = split.split("@")
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # Load Json
        # if split in ['train', 'val_seen', 'val_unseen', 'test',
        #              'val_unseen_half1', 'val_unseen_half2', 'val_seen_half1', 'val_seen_half2']:       # Add two halves for sanity check
        if "/" not in split:
            with open('data/R2R_%s.json' % split) as f:
                new_data = json.load(f)
        else:
            print('\nLoading prevalent data for pretraining...')
            with open(split) as f:
                new_data = json.load(f)

        # Partition
        if number > 0:
            random.seed(1)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    return data


def load_my_datasets(splits):
    """

    :param splits: A list of split.
        if the split is "something@5000", it will use a random 5000 data from the data
    :return:
    """
    import random
    data = []
    old_state = random.getstate()
    for split in splits:
        # It only needs some part of the dataset?
        components = split.split("@")
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # Load Json
        # if split in ['train', 'val_seen', 'val_unseen', 'test',
        #              'val_unseen_half1', 'val_unseen_half2', 'val_seen_half1', 'val_seen_half2']:       # Add two halves for sanity check
        if "/" not in split:
            with open('data/airvlnce/AIRVINCE_%s.json' % split) as f:
                new_data = json.load(f)
        else:
            raise NotImplementedError()
            print('\nLoading prevalent data for pretraining...')
            with open(split) as f:
                new_data = json.load(f)

        # Partition
        if number > 0:
            random.seed(1)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    return data


def pad_instr_tokens(instr_tokens, maxlength=20):

    if len(instr_tokens) <= 2: #assert len(raw_instr_tokens) > 2
        return None

    if len(instr_tokens) > maxlength - 2: # -2 for [CLS] and [SEP]
        instr_tokens = instr_tokens[:(maxlength-2)]

    instr_tokens = ['[CLS]'] + instr_tokens + ['[SEP]']
    num_words = len(instr_tokens)  # - 1  # include [SEP]
    instr_tokens += ['[PAD]'] * (maxlength-len(instr_tokens))

    assert len(instr_tokens) == maxlength

    return instr_tokens, num_words


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, vocab=None, encoding_length=300):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        self.index_to_word = {}

        ori_vocab_length = 0

        if vocab:
            ori_vocab_length = len(vocab)

            for i,word in enumerate(vocab):
                self.word_to_index[word] = i
            new_w2i = defaultdict(lambda: self.word_to_index['<UNK>'])
            new_w2i.update(self.word_to_index)
            self.word_to_index = new_w2i
            for key, value in self.word_to_index.items():
                self.index_to_word[value] = key

        old = self.vocab_size()
        logger.info("OLD_VOCAB_SIZE: {}".format(old))

        assert len(self.word_to_index) == ori_vocab_length, 'error'

        if vocab:
            logger.info("VOACB: {}".format(len(vocab)))
        else:
            logger.info("VOACB: {}".format(0))

    def finalize(self):
        """
        This is used for debug
        """
        self.word_to_index = dict(self.word_to_index)   # To avoid using mis-typing tokens

    def add_word(self, word):
        assert word not in self.word_to_index
        self.word_to_index[word] = self.vocab_size()    # vocab_size() is the
        self.index_to_word[self.vocab_size()] = word

    @staticmethod
    def split_sentence(sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def vocab_size(self):
        return len(self.index_to_word)

    def encode_sentence(self, sentence, max_length=None):
        if max_length is None:
            max_length = self.encoding_length
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = []
        for word in self.split_sentence(sentence):
            if word in self.word_to_index.keys():
                encoding.append(self.word_to_index[word])  # Default Dict
            else:
                encoding.append(self.word_to_index['<UNK>'])

        if len(encoding) <= 0:
            return None
        #assert len(encoding) > 2

        if len(encoding) < max_length:
            encoding += [self.word_to_index['<PAD>']] * (max_length-len(encoding))  # Padding
        elif len(encoding) > max_length:
            pass

        return np.array(encoding[:max_length])

    def decode_sentence(self, encoding, length=None):
        sentence = []
        if length is not None:
            encoding = encoding[:length]
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.index_to_word[ix])
        return " ".join(sentence)

    def shrink(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <EOS> return empty list
        """
        raise  NotImplementedError

        if len(inst) == 0:
            return inst
        end = np.argmax(np.array(inst) == self.word_to_index['<EOS>'])     # If no <EOS>, return empty string
        if len(inst) > 1 and inst[0] == self.word_to_index['<BOS>']:
            start = 1
        else:
            start = 0
        # print(inst, start, end)
        return inst[start: end]


def build_vocab(splits=['train'], min_count=1, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_my_datasets(splits)
    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def read_img_features(feature_store, test_only=False):
    import csv
    import base64

    print("Start loading the image feature ... (~50 seconds)")
    start = time.time()

    if "detectfeat" in args.features:
        views = int(args.features[10:])
    else:
        views = 36

    args.views = views

    tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']

    if not test_only:
        features = {}
        with open(feature_store, "r") as tsv_in_file:     # Open the tsv file.
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
            for item in reader:
                long_id = item['scanId'] + "_" + item['viewpointId']
                features[long_id] = np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                                   dtype=np.float32).reshape((views, -1))   # Feature of long_id is (36, 2048)
    else:
        features = None

    print("Finish Loading the image feature from %s in %0.4f seconds" % (feature_store, time.time() - start))
    return features

def read_candidates(candidates_store):
    import csv
    import base64
    from collections import defaultdict
    print("Start loading the candidate feature")

    start = time.time()

    TSV_FIELDNAMES = ['scanId', 'viewpointId', 'heading', 'elevation', 'next', 'pointId', 'idx', 'feature']
    candidates = defaultdict(lambda: list())
    items = 0
    with open(candidates_store, "r") as tsv_in_file:     # Open the tsv file.
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=TSV_FIELDNAMES)
        for item in reader:
            long_id = item['scanId'] + "_" + item['viewpointId']
            candidates[long_id].append(
                {'heading': float(item['heading']),
                 'elevation': float(item['elevation']),
                 'scanId': item['scanId'],
                 'viewpointId': item['next'],
                 'pointId': int(item['pointId']),
                 'idx': int(item['idx']) + 1,   # Because a bug in the precompute code, here +1 is important
                 'feature': np.frombuffer(
                     base64.decodestring(item['feature'].encode('ascii')),
                     dtype=np.float32)
                    }
            )
            items += 1

    for long_id in candidates:
        assert (len(candidates[long_id])) != 0

    assert sum(len(candidate) for candidate in candidates.values()) == items

    # candidate = candidates[long_id]
    # print(candidate)
    print("Finish Loading the candidates from %s in %0.4f seconds" % (candidates_store, time.time() - start))
    candidates = dict(candidates)
    return candidates

def add_exploration(paths):
    explore = json.load(open("data/exploration.json", 'r'))
    inst2explore = {path['instr_id']: path['trajectory'] for path in explore}
    for path in paths:
        path['trajectory'] = inst2explore[path['instr_id']] + path['trajectory']
    return paths

def angle_feature(heading, elevation):

    import math
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation)] * (args.angle_feat_size // 4),
                    dtype=np.float32)

def new_simulator():
    import MatterSim
    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.init()

    return sim

def get_point_angle_feature(baseViewId=0):
    sim = new_simulator()

    feature = np.empty((36, args.angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            sim.newEpisode('ZMojNkEp431', '2f4d90acd4024c269fb0efe49a8ac540', 0, math.radians(-30))
        elif ix % 12 == 0:
            sim.makeAction(0, 1.0, 1.0) # 转了一圈
        else:
            sim.makeAction(0, 1.0, 0) # 向右旋转

        state = sim.getState()
        assert state.viewIndex == ix

        heading = state.heading - base_heading

        feature[ix, :] = angle_feature(heading, state.elevation)
    return feature

def get_all_point_angle_feature():
    return [get_point_angle_feature(baseViewId) for baseViewId in range(36)]

def add_idx(inst):
    toks = Tokenizer.split_sentence(inst)
    return " ".join([str(idx)+tok for idx, tok in enumerate(toks)])

import signal
class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

from collections import OrderedDict

class Timer:
    def __init__(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def reset(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def tic(self, key):
        self.start[key] = time.time()

    def toc(self, key):
        delta = time.time() - self.start[key]
        if key not in self.cul:
            self.cul[key] = delta
        else:
            self.cul[key] += delta

    def step(self):
        self.iter += 1

    def show(self):
        total = sum(self.cul.values())
        for key in self.cul:
            print("%s, total time %0.2f, avg time %0.2f, part of %0.2f" %
                  (key, self.cul[key], self.cul[key]*1./self.iter, self.cul[key]*1./total))
        print(total / self.iter)


stop_word_list = [
    ",", ".", "and", "?", "!"
]


def stop_words_location(inst, mask=False):
    toks = Tokenizer.split_sentence(inst)
    sws = [i for i, tok in enumerate(toks) if tok in stop_word_list]        # The index of the stop words
    if len(sws) == 0 or sws[-1] != (len(toks)-1):     # Add the index of the last token
        sws.append(len(toks)-1)
    sws = [x for x, y in zip(sws[:-1], sws[1:]) if x+1 != y] + [sws[-1]]    # Filter the adjacent stop word
    sws_mask = np.ones(len(toks), np.int32)         # Create the mask
    sws_mask[sws] = 0
    return sws_mask if mask else sws

def get_segments(inst, mask=False):
    toks = Tokenizer.split_sentence(inst)
    sws = [i for i, tok in enumerate(toks) if tok in stop_word_list]        # The index of the stop words
    sws = [-1] + sws + [len(toks)]      # Add the <start> and <end> positions
    segments = [toks[sws[i]+1:sws[i+1]] for i in range(len(sws)-1)]       # Slice the segments from the tokens
    segments = list(filter(lambda x: len(x)>0, segments))     # remove the consecutive stop words
    return segments

def clever_pad_sequence(sequences, batch_first=True, padding_value=0):
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    max_len = max(seq.size()[0] for seq in sequences)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    if padding_value is not None:
        out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

import torch
def length2mask(length, size=None):
    batch_size = len(length)
    size = 1 if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask

def average_length(path2inst):
    length = []

    for name in path2inst:
        datum = path2inst[name]
        length.append(len(datum))
    return sum(length) / len(length)

def tile_batch(tensor, multiplier):
    _, *s = tensor.size()
    tensor = tensor.unsqueeze(1).expand(-1, multiplier, *(-1,) * len(s)).contiguous().view(-1, *s)
    return tensor

def viewpoint_drop_mask(viewpoint, seed=None, drop_func=None):
    local_seed = hash(viewpoint) ^ seed
    torch.random.manual_seed(local_seed)
    drop_mask = drop_func(torch.ones(2048).cuda())
    return drop_mask


class FloydGraph:
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def ndtw_initialize():
    ndtw_criterion = {}
    scan_gts_dir = 'data/id_paths.json'
    with open(scan_gts_dir) as f_:
        scan_gts = json.load(f_)
    all_scan_ids = []
    for key in scan_gts:
        path_scan_id = scan_gts[key][0]
        # print('path_scan_id', path_scan_id)
        if path_scan_id not in all_scan_ids:
            all_scan_ids.append(path_scan_id)
            ndtw_graph = ndtw_graphload(path_scan_id)
            ndtw_criterion[path_scan_id] = DTW(ndtw_graph)
    return ndtw_criterion

def ndtw_graphload(scan):
    """Loads a networkx graph for a given scan.
    Args:
    connections_file: A string with the path to the .json file with the
      connectivity information.
    Returns:
    A networkx graph.
    """
    connections_file = 'connectivity/{}_connectivity.json'.format(scan)
    with open(connections_file) as f:
        lines = json.load(f)
        nodes = np.array([x['image_id'] for x in lines])
        matrix = np.array([x['unobstructed'] for x in lines])
        mask = np.array([x['included'] for x in lines])

        matrix = matrix[mask][:, mask]
        nodes = nodes[mask]

        pos2d = {x['image_id']: np.array(x['pose'])[[3, 7]] for x in lines}
        pos3d = {x['image_id']: np.array(x['pose'])[[3, 7, 11]] for x in lines}

    graph = nx.from_numpy_matrix(matrix)
    graph = nx.relabel.relabel_nodes(graph, dict(enumerate(nodes)))
    nx.set_node_attributes(graph, pos2d, 'pos2d')
    nx.set_node_attributes(graph, pos3d, 'pos3d')

    weight2d = {(u, v): norm(pos2d[u] - pos2d[v]) for u, v in graph.edges}
    weight3d = {(u, v): norm(pos3d[u] - pos3d[v]) for u, v in graph.edges}
    nx.set_edge_attributes(graph, weight2d, 'weight2d')
    nx.set_edge_attributes(graph, weight3d, 'weight3d')

    return graph

class DTW(object):
  """Dynamic Time Warping (DTW) evaluation metrics.
  Python doctest:
  >>> graph = nx.grid_graph([3, 4])
  >>> prediction = [(0, 0), (1, 0), (2, 0), (3, 0)]
  >>> reference = [(0, 0), (1, 0), (2, 1), (3, 2)]
  >>> dtw = DTW(graph)
  >>> assert np.isclose(dtw(prediction, reference, 'dtw'), 3.0)
  >>> assert np.isclose(dtw(prediction, reference, 'ndtw'), 0.77880078307140488)
  >>> assert np.isclose(dtw(prediction, reference, 'sdtw'), 0.77880078307140488)
  >>> assert np.isclose(dtw(prediction[:2], reference, 'sdtw'), 0.0)
  """

  def __init__(self, graph, weight='weight', threshold=3.0):
    """Initializes a DTW object.
    Args:
      graph: networkx graph for the environment.
      weight: networkx edge weight key (str).
      threshold: distance threshold $d_{th}$ (float).
    """
    self.graph = graph
    self.weight = weight
    self.threshold = threshold
    self.distance = dict(
        nx.all_pairs_dijkstra_path_length(self.graph, weight=self.weight))

  def __call__(self, prediction, reference, metric='sdtw'):
    """Computes DTW metrics.
    Args:
      prediction: list of nodes (str), path predicted by agent.
      reference: list of nodes (str), the ground truth path.
      metric: one of ['ndtw', 'sdtw', 'dtw'].
    Returns:
      the DTW between the prediction and reference path (float).
    """
    assert metric in ['ndtw', 'sdtw', 'dtw']

    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction)+1):
      for j in range(1, len(reference)+1):
        best_previous_cost = min(
            dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
        cost = self.distance[prediction[i-1]][reference[j-1]]
        dtw_matrix[i][j] = cost + best_previous_cost
    dtw = dtw_matrix[len(prediction)][len(reference)]

    if metric == 'dtw':
      return dtw

    ndtw = np.exp(-dtw/(self.threshold * len(reference)))
    if metric == 'ndtw':
      return ndtw

    success = self.distance[prediction[-1]][reference[-1]] <= self.threshold
    return success * ndtw
