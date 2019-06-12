from optimizer.util.xmlmodifier import XmlModifier

m = XmlModifier("../data/fair-scheduler.xml", "../results/test.xml")
m.modify_property('queueA', 1)
m.save()
