import torch

class DeviceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance.device = 'cuda'
        return cls._instance

    def set_device(self, device_name):
        self.device = device_name

    def get_device(self):
        return self.device


class DatetypeManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatetypeManager, cls).__new__(cls)
            cls._instance.datetype = torch.complex64
        return cls._instance

    def set_datetype(self, date_type):
        self.datetype = date_type

    def get_datetype(self):
        return self.datetype