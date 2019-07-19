import json
import random

import math

from workloadsubmission.datamodel import alg
from workloadsubmission.datamodel import dataset


def interval(mean):
    p = 1.0
    k = 0
    e = math.exp(-mean)
    while p >= e:
        u = random.random()
        p *= u
        k += 1
    return k


def getitem(load_dict):
    setting = dataset()
    for key in load_dict.keys():
        item = alg()
        item.interval = load_dict[key]["interval"]
        item.queue = load_dict[key]["queue"]
        item.name = load_dict[key]["name"]
        item.data_size = load_dict[key]["datasize"]
        setting.item.append(item)
        setting.length += 1
    return setting


def getjson(jsonstr):
    setting = dataset()
    # with open(path_str,"r") as load_f:
    #     load_dict = json.load(load_f)
    # load_dict = json.dumps(jsonstr, default=dataset)
    load_dict = json.loads(jsonstr, encoding='UTF-8')
    # print(load_dict)
    # setting.queue = load_dict['queue']
    # setting.item.append(load_dict.)
    setting.algorithm[1] = load_dict['algorithm1']
    setting.algorithm[2] = load_dict['algorithm2']
    setting.algorithm[3] = load_dict['algorithm3']
    setting.algorithm[4] = load_dict['algorithm4']
    setting.algorithm[5] = load_dict['algorithm5']
    setting.algorithm[6] = load_dict['algorithm6']
    setting.algorithm[7] = load_dict['algorithm7']

    setting.algorithm_size[1] = load_dict['algorithm1_size']
    setting.algorithm_size[2] = load_dict['algorithm2_size']
    setting.algorithm_size[3] = load_dict['algorithm3_size']
    setting.algorithm_size[4] = load_dict['algorithm4_size']
    setting.algorithm_size[5] = load_dict['algorithm5_size']
    setting.algorithm_size[6] = load_dict['algorithm6_size']
    setting.algorithm_size[7] = load_dict['algorithm7_size']

    setting.algorithm_queue[1] = load_dict['algorithm1_queue']
    setting.algorithm_queue[2] = load_dict['algorithm2_queue']
    setting.algorithm_queue[3] = load_dict['algorithm3_queue']
    setting.algorithm_queue[4] = load_dict['algorithm4_queue']
    setting.algorithm_queue[5] = load_dict['algorithm5_queue']
    setting.algorithm_queue[6] = load_dict['algorithm6_queue']
    setting.algorithm_queue[7] = load_dict['algorithm7_queue']

    setting.algorithm_interval[1] = load_dict['algorithm1_interval']
    setting.algorithm_interval[2] = load_dict['algorithm2_interval']
    setting.algorithm_interval[3] = load_dict['algorithm3_interval']
    setting.algorithm_interval[4] = load_dict['algorithm4_interval']
    setting.algorithm_interval[5] = load_dict['algorithm5_interval']
    setting.algorithm_interval[6] = load_dict['algorithm6_interval']
    setting.algorithm_interval[7] = load_dict['algorithm7_interval']

    # setting.interval = interval(load_dict['interval'])
    # print(setting.algorithm1)
    return setting
