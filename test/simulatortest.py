from optimizer.environment.delayprediction.resourceallocationsimulator import ResourceAllocationSimulator

simulator = ResourceAllocationSimulator()
simulator.set_resources([5, 2])
simulator.release(5, 0, 2)
simulator.release(15, 1, 1)
simulator.release(20, 1, 3)
simulator.release(10, 0, 6)
simulator.release(6, 0, 5)

a = simulator.allocate(0, 30)
print(a)
simulator.release(50, 0, 30)

b = simulator.allocate(1, 300)

print(b)

c = simulator.allocate(0, 30)
print(c)
