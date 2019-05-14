from bs4 import BeautifulSoup


class XmlModifier(object):
    """
    A XML modifier that helps to modify key-value value of XML.
    """

    def __init__(self, from_filename: str, to_filename: str):
        self.from_filename = from_filename
        self.to_filename = to_filename
        with open(self.from_filename, 'r') as f:
            text = f.read()
            self.data = BeautifulSoup(text, features='lxml-xml')

    def modify(self, key, value):
        names = self.data.find_all('name')
        for n in names:
            if n.string == key:
                v = n.next_sibling.next_sibling
                v.string = str(value)
                return

    def save(self):
        with open(self.to_filename, 'w') as f:
            f.write(self.data.prettify())
