STATE_SHAPE = (200, 200)

PRE_TRAIN_LOOP_INTERNAL = 5
TRAIN_LOOP_INTERNAL = 1
TEST_LOOP_INTERNAL = 5
EVALUATION_LOOP_INTERNAL = 10

QUEUES = {
    "names": ["queueA", "queueB"],
    "actions": {
        0: [1, 5],
        1: [2, 4],
        2: [3, 3],
        3: [4, 2],
        4: [5, 1]
    }
}
