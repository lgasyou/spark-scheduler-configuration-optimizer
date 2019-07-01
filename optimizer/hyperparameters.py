STATE_SHAPE = (200, 200)

PRE_TRAIN_LOOP_INTERNAL = 5
TRAIN_LOOP_INTERNAL = 1
TEST_LOOP_INTERNAL = 5
EVALUATION_LOOP_INTERNAL = 10

QUEUES = {
    "names": ["queueA", "queueB", "queueC", "queueD"],
    "actions": {
        0: [1, 1, 1, 3],
        1: [1, 1, 2, 2],
        2: [1, 1, 3, 1],
        3: [1, 2, 1, 2],
        4: [1, 2, 2, 1],
        5: [1, 3, 1, 1],
        6: [2, 1, 1, 2],
        7: [2, 1, 2, 1],
        8: [2, 2, 1, 1],
        9: [3, 1, 1, 1]
    }
}
