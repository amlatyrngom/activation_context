"""
Small shared helpers with no dependencies on the rest of the package.
"""
from __future__ import annotations

import importlib
import sys
import typing as t


class MissingClassError(RuntimeError):
    """A class named by a record (a tool, an env setup, an agentic program) is not importable here."""


class MissingClass:
    """
    Stands in for a class spec that did not resolve when a record was read leniently: it keeps the module and qualified
    name the record named (so re-serializing the record leaves the spec unchanged), and it refuses construction with a
    MissingClassError that names the class. Training and analysis read such records as usual; anything that would run
    them fails at the point of construction.
    """

    def __init__(self, module: str, qualname: str, error: str):
        self.__module__ = module
        self.__qualname__ = qualname
        self.__name__ = qualname.rsplit(".", 1)[-1]
        self.error = error

    def __call__(self, *args, **kwargs):
        raise MissingClassError(f"{self.__module__}:{self.__qualname__} is not importable here ({self.error}); "
                                "the record can be read and trained on, not run")

    def __repr__(self) -> str:
        return f"MissingClass({self.__module__}:{self.__qualname__})"


def class_spec_name(cls: t.Any) -> str:
    """
    "module:Qualname" for a class (or a MissingClass). A class defined in a script run as `python -m pkg.script` reports
    module `__main__`; that is mapped to the script's real module name, so records written by study scripts stay importable.
    """
    module = cls.__module__
    if module == "__main__":
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        if spec is not None and getattr(spec, "name", None):
            module = spec.name
    return f"{module}:{cls.__qualname__}"


def resolve_class_name(name: str, strict: bool = True) -> t.Any:
    """
    The class named "module:Qualname". With `strict`, an import or attribute failure raises; without, it returns a
    MissingClass placeholder carrying the name and the failure, for records read without the class present.
    """
    module_name, _, qualname = name.partition(":")
    try:
        cls: t.Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        return cls
    except Exception as error:
        if strict:
            raise
        return MissingClass(module_name, qualname, f"{type(error).__name__}: {error}")
