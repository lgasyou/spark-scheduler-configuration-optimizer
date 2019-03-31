import unittest

from src.xmlutil import XmlModifier


class XmlUtilTest(unittest.TestCase):

    def test(self):
        mod = XmlModifier('../data/capacity-scheduler-template.xml', '../results/capacity-scheduler.xml')
        mod.modify('yarn.scheduler.capacity.root.queueA.capacity', 12)
        mod.modify('yarn.scheduler.capacity.root.queueB.capacity', 100)
        mod.save()
