STATE_SHAPE = (5042, 5000)

PRE_TRAIN_LOOP_INTERNAL = 5
TRAIN_LOOP_INTERNAL = 180
TEST_LOOP_INTERNAL = 5
EVALUATION_LOOP_INTERNAL = 10

QUEUES = {
    "names": ["QueueA", "QueueB"],
    "actions": {
        0: [1, 5],
        1: [2, 4],
        2: [3, 3],
        3: [4, 2],
        4: [5, 1]
    }
}
