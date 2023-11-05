from functools import lru_cache
from hashlib import md5
from pprint import pprint


class ConfigBase:
    @classmethod
    @lru_cache(maxsize=1)
    def to_dict(cls):
        keys = dir(cls)
        hp_dict = {}
        for key in keys:
            value = getattr(cls, key)
            if (
                not key.startswith("__")
                and not key.endswith("__")
                and not callable(value)
            ):
                hp_dict[key] = value
        return hp_dict

    @classmethod
    @lru_cache(maxsize=1)
    def to_md5(cls):
        hp_dict = cls.to_dict()
        return md5(f"{hp_dict}".encode("utf-8")).hexdigest()

    @classmethod
    @lru_cache(maxsize=1)
    def instance(cls):
        return cls()

    @classmethod
    def print(cls):
        print(f"hash (md5): {cls.to_md5()}")
        pprint(cls.to_dict())
