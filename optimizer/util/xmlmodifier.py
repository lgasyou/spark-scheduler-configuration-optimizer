from bs4 import BeautifulSoup


class XmlModifier(object):
    """
    A XML modifier who helps to modify the value of key-value type in XML file.
    """

    def __init__(self, from_filename: str, to_filename: str):
        self.from_filename = from_filename
        self.to_filename = to_filename
        with open(self.from_filename, 'r') as f:
            text = f.read()
            self.data = BeautifulSoup(text, features='lxml-xml')

    def modify_kv_type(self, key, value):
        names = self.data.find_all('name')
        for n in names:
            if n.string == key:
                v = n.next_sibling.next_sibling
                v.string = str(value)
                return

    def modify_property(self, name, pro):
        names = self.data.find_all('queue')
        for n in names:
            if n['name'] == name:
                n.weight.string = str(pro)
                return

    def save(self):
        with open(self.to_filename, 'w') as f:
            f.write(self.data.prettify())
