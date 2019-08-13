#!/usr/bin/env python


from abc import ABC, abstractmethod


class InferenceModel(ABC):

    @abstractmethod
    def predict(self):
        pass
