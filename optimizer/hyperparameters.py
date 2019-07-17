CUDA_DEVICES = [5, 6, 7]

STATE_SHAPE = (200, 200)

PRE_TRAIN_LOOP_INTERNAL = 5
TRAIN_LOOP_INTERNAL = 180
TEST_LOOP_INTERNAL = 5
EVALUATION_LOOP_INTERNAL = 10

QUEUES = {
    "names": ["queueA", "queueB", "queueC", "queueD"],
    "actions": {
        "CapacityScheduler": {
            0: ([20, 20, 20, 40], [80, 80, 80, 80]),
            1: ([20, 20, 30, 30], [80, 80, 80, 80]),
            2: ([20, 20, 40, 20], [80, 80, 80, 80]),
            3: ([20, 30, 20, 30], [80, 80, 80, 80]),
            4: ([20, 30, 30, 20], [80, 80, 80, 80]),
            5: ([20, 40, 20, 20], [80, 80, 80, 80]),
            6: ([30, 20, 20, 30], [80, 80, 80, 80]),
            7: ([30, 20, 30, 20], [80, 80, 80, 80]),
            8: ([30, 30, 20, 20], [80, 80, 80, 80]),
            9: ([40, 20, 20, 20], [80, 80, 80, 80]),
        },
        "FairScheduler": {
            0: ([1, 1, 1, 3], 'fair'),
            1: ([1, 1, 2, 2], 'fair'),
            2: ([1, 1, 3, 1], 'fair'),
            3: ([1, 2, 1, 2], 'fair'),
            4: ([1, 2, 2, 1], 'fair'),
            5: ([1, 3, 1, 1], 'fair'),
            6: ([2, 1, 1, 2], 'fair'),
            7: ([2, 1, 2, 1], 'fair'),
            8: ([2, 2, 1, 1], 'fair'),
            9: ([3, 1, 1, 1], 'fair'),

            10: ([1, 1, 1, 3], 'fifo'),
            11: ([1, 1, 2, 2], 'fifo'),
            12: ([1, 1, 3, 1], 'fifo'),
            13: ([1, 2, 1, 2], 'fifo'),
            14: ([1, 2, 2, 1], 'fifo'),
            15: ([1, 3, 1, 1], 'fifo'),
            16: ([2, 1, 1, 2], 'fifo'),
            17: ([2, 1, 2, 1], 'fifo'),
            18: ([2, 2, 1, 1], 'fifo'),
            19: ([3, 1, 1, 1], 'fifo'),

            20: ([1, 1, 1, 3], 'drf'),
            21: ([1, 1, 2, 2], 'drf'),
            22: ([1, 1, 3, 1], 'drf'),
            23: ([1, 2, 1, 2], 'drf'),
            24: ([1, 2, 2, 1], 'drf'),
            25: ([1, 3, 1, 1], 'drf'),
            26: ([2, 1, 1, 2], 'drf'),
            27: ([2, 1, 2, 1], 'drf'),
            28: ([2, 2, 1, 1], 'drf'),
            29: ([3, 1, 1, 1], 'drf'),
        }
    }
}
