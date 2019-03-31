class XmlModifier(object):
    """
    A XML modifier that helps to modify key-value value of XML.
    """

    def __init__(self, from_filename: str, to_filename: str):
        self.from_filename = from_filename
        self.to_filename = to_filename
        with open(self.from_filename, 'r') as f:
            self.data = f.read()

    # TODO: Implementation
    def modify(self, key, value):
        pass

    def save(self):
        with open(self.to_filename, 'w') as f:
            f.write(self.data)
