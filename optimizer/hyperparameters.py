CUDA_DEVICES = [i for i in range(10)]

STATE_SHAPE = (200, 200)

WAIT_CONFIG_APPLY_TIME = 3

TRAINING_LOOP_INTERNAL = 3
OPTIMIZATION_LOOP_INTERNAL = 5
EVALUATION_LOOP_INTERNAL = 5

TRAINING_LOOP_INTERNAL -= WAIT_CONFIG_APPLY_TIME
OPTIMIZATION_LOOP_INTERNAL -= WAIT_CONFIG_APPLY_TIME
EVALUATION_LOOP_INTERNAL -= WAIT_CONFIG_APPLY_TIME
if TRAINING_LOOP_INTERNAL < 0 or OPTIMIZATION_LOOP_INTERNAL < 0 or EVALUATION_LOOP_INTERNAL < 0:
    raise RuntimeError('Interval is less than 0.')

QUEUES = {
    "names": ["queueA", "queueB"],
    "actions": {
        "capacityScheduler": {
            0: ([25, 75], [40, 80]),
            1: ([25, 75], [60, 80]),
            2: ([25, 75], [80, 80]),

            3: ([50, 50], [50, 50]),
            4: ([50, 50], [60, 60]),
            5: ([50, 50], [80, 80]),

            6: ([75, 25], [80, 40]),
            7: ([75, 25], [80, 60]),
            8: ([75, 25], [80, 80]),
        },
        "fairScheduler": {
            0: ([25, 25, 75, 75], 'fifo'),
            1: ([25, 25, 75, 75], 'fair'),
            2: ([25, 25, 75, 75], 'drf'),

            3: ([25, 75, 25, 75], 'fifo'),
            4: ([25, 75, 25, 75], 'fair'),
            5: ([25, 75, 25, 75], 'drf'),

            6: ([25, 75, 75, 25], 'fifo'),
            7: ([25, 75, 75, 25], 'fair'),
            8: ([25, 75, 75, 25], 'drf'),

            9: ([75, 25, 25, 75], 'fifo'),
            10: ([75, 25, 25, 75], 'fair'),
            11: ([75, 25, 25, 75], 'drf'),

            12: ([75, 25, 75, 25], 'fifo'),
            13: ([75, 25, 75, 25], 'fair'),
            14: ([75, 25, 75, 25], 'drf'),

            15: ([75, 75, 25, 25], 'fifo'),
            16: ([75, 75, 25, 25], 'fair'),
            17: ([75, 75, 25, 25], 'drf'),

            18: ([50, 50, 50, 50], 'fifo'),
            19: ([50, 50, 50, 50], 'fair'),
            20: ([50, 50, 50, 50], 'drf'),
        }
    }
}
