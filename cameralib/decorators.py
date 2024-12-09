import functools
import numpy as np

def point_transform(f):
    """Decorator to make a function, which transforms multiple points, also accept a single point,
    as well as lists, tuples etc. that can be converted by np.asarray."""

    @functools.wraps(f)
    def wrapped(self, points, *args, **kwargs):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 2:
            return f(self, points, *args, **kwargs)

        reshaped = np.reshape(points, [-1, points.shape[-1]])
        reshaped_result = f(self, reshaped, *args, **kwargs)
        return np.reshape(reshaped_result, [*points.shape[:-1], -1])

    return wrapped


def camera_transform(f):
    """Decorator to make a function, which transforms the camera,
    also accept an 'inplace' argument."""

    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        inplace = kwargs.pop('inplace', True)
        if inplace:
            f(self, *args, **kwargs)
            return self
        else:
            camcopy = self.copy()
            f(camcopy, *args, **kwargs)
            return camcopy

    return wrapped