import pandas as pd

for i in range(3):
    with open('../results/no-optim-time-delays-%d-0.txt' % i, 'r') as f:
        d: dict = eval(f.readline())
        summary = []
        for k, v in d.items():
            summary.append([k * 3, v / 1000])

        print(summary)
        writer = pd.ExcelWriter('../results/固定配置项%d的时延.xlsx' % i)
        df = pd.DataFrame(summary, columns=['时间（分钟）', '三分钟内完成作业的平均时延（秒）'])
        df.to_excel(writer)
        writer.save()

with open('../results/optim-time-delays-0.txt', 'r') as f:
    d: dict = eval(f.readline())
    summary = []
    for k, v in d.items():
        summary.append([k * 3, v / 1000])

    print(summary)
    writer = pd.ExcelWriter('../results/优化时延.xlsx')
    df = pd.DataFrame(summary, columns=['时间（分钟）', '三分钟内完成作业的平均时延（秒）'])
    df.to_excel(writer)
    writer.save()
