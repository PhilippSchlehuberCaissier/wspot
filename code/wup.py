from dataclasses import dataclass


# We write the value of wup to a file in /tmp
# TODO this is quite unelegant
# TODO also this is not portable (no /tmp on Windows)
class ExtractWUP:
    def __get__(self, obj, objtype=None):
        f = open("/tmp/wup", 'r')
        return int(f.readline())

@dataclass
class WUP:
    value = ExtractWUP()


def set_wup(wup):
    with open("/tmp/wup", 'w') as f:
        print(wup, file=f)
