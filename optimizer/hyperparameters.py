CUDA_DEVICES = [0]

STATE_SHAPE = (20, 4)

CONTAINER_NUM_VCORES = 4
CONTAINER_MEM = 8

TIME_STEP_INTERVAL = 20
WAIT_CONFIG_APPLY_TIME = 15
EXTRA_WAIT_TIME = TIME_STEP_INTERVAL - WAIT_CONFIG_APPLY_TIME

if EXTRA_WAIT_TIME < 0:
    raise RuntimeError('Interval is less than 0.')

QUEUES = {
    "names": ["queueA", "queueB"],
    "actions": {
        "capacityScheduler": {
            0: ([3, 7], [6, 8]),
            1: ([5, 5], [5, 5]),
            2: ([7, 3], [8, 8]),
        },
        "fairScheduler": {
            0: ([25, 75], 'fifo'),
            1: ([25, 75], 'fair'),
            2: ([25, 75], 'drf'),

            3: ([50, 50], 'fifo'),
            4: ([50, 50], 'fair'),
            5: ([50, 50], 'drf'),

            6: ([75, 25], 'fifo'),
            7: ([75, 25], 'fair'),
            8: ([75, 25], 'drf'),
        }
    }
}
