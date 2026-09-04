import difflib


_DEFAULTS = {"left", "right", "top", "bottom", "front", "back"}


class TagKeyError(KeyError):

    def __str__(self):
        return self.args[0]


class TagDict(dict):

    _kind = "tag"
    _ctor = "Mesh.with_boundaries"

    def __missing__(self, key):
        raise TagKeyError(self._message(key))

    def _message(self, key):
        out = [f"{self._kind} tag {key!r} is not defined on this mesh."]
        if not self:
            out.append(f"No {self._kind} tags are defined.")
            if self._kind == "boundary" and str(key) in _DEFAULTS:
                out.append(
                    "Since scikit-fem 10.0.0 the default tags ('left', "
                    "'right', 'top', ...) are no longer added automatically:\n"
                    "    mesh = mesh.with_defaults()")
            else:
                out.append(f"Define tags with {self._ctor}(...).")
        else:
            keys = sorted(map(str, self))
            near = difflib.get_close_matches(str(key), keys, n=3, cutoff=0.6)
            if near:
                out.append("Did you mean: "
                           + ", ".join(repr(k) for k in near) + "?")
            out.append("Defined: " + ", ".join(repr(k) for k in keys))
        return "\n".join(out)


class Boundaries(TagDict):
    _kind, _ctor = "boundary", "Mesh.with_boundaries"


class Subdomains(TagDict):
    _kind, _ctor = "subdomain", "Mesh.with_subdomains"
