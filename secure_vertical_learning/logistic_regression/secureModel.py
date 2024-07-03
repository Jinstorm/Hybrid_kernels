class SecureLRModel(object):
    def __init__(self, modelName) -> None:
        self.modelName = modelName

    def getModelinfo(self):
        print(self.modelName)
        return self.modelName