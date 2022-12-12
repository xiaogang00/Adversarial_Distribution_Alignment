
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader_cifar10 import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
