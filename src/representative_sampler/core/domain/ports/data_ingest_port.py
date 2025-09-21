
from abc import ABC, abstractmethod


class DataIngestPort(ABC):
    @abstractmethod
    def ingest_data(self, data):
        raise NotImplementedError("This method should be overridden by subclasses.")