from optimizer.environment import actionparsing
from optimizer.util.xmlmodifier import XmlModifier


m = XmlModifier('../data/fair-scheduler-template.xml', '../results/test.xml')
m.modify_all_values('schedulingPolicy', 'fifo')
m.save()


print(actionparsing.parse('FairScheduler'))
