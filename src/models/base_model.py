from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def forward(self, **inputs):
        pass
        
    @abstractmethod
    def generate(self, **inputs):
        pass 