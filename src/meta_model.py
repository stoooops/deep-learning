#!/usr/bin/env python


class MetaModel:

    def __init__(self, name, keras_model=None, tflite_interpretter=None):
        self.name = name
        self.keras_model = keras_model
        self.tflite_interpretter = tflite_interpretter

    def compile(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.evaluate(*args, **kwargs)

    def save(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.save(*args, **kwargs)

    def summary(self, *args, **kwargs):
        assert self.keras_model is not None
        return self.keras_model.summary(*args, **kwargs)
