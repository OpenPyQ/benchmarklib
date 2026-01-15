from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable, Union

from sqlalchemy import Boolean, Column, ForeignKey, Index, Integer, String, JSON, Float, DateTime, LargeBinary, select, create_engine, select, func, or_, text
from sqlalchemy.orm import  declared_attr, Mapped, relationship, mapped_column, sessionmaker, Session, DeclarativeBase, DeclarativeMeta, Mapped, relationship, selectinload, joinedload
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.ext.hybrid import hybrid_property

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import *
from qiskit.passmanager.flow_controllers import FlowControllerLinear, DoWhileController, ConditionalController, BaseController
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend


from benchmarklib.core.base import Base
from .registries import PassManagerFactoryRegistry, SynthesizerRegistry, StepRegistry


class PipelineConfig(Base):
    __tablename__ = "pipeline_configs"

    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    synthesizer_config: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)
    compiler_config: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)
    pass_manager_factory_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    pass_manager_kwargs: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)
    backend_name: Mapped[Optional[str]]

    def __init__(self, *args, **kwargs):
        factory = kwargs.pop("pass_manager_factory", None)
        backend = kwargs.pop("backend", None)

        super().__init__(*args, **kwargs)

        if factory is not None:
            self.pass_manager_factory = factory
        if backend is not None:
            self.backend = backend

    @hybrid_property
    def backend(self):
        return QiskitRuntimeService().backend(self.backend_name) if self.backend_name else None
    
    @backend.setter
    def backend(self, backend: Optional[IBMBackend]):
        self.backend_name = backend.name if backend else None
    
    @hybrid_property
    def pass_manager_factory(self):
        return PassManagerFactoryRegistry.get(self.pass_manager_factory_name) if self.pass_manager_factory_name else None
    
    @pass_manager_factory.setter
    def pass_manager_factory(self, factory: Optional[Callable[..., PassManager]]):
        self.pass_manager_factory_name = PassManagerFactoryRegistry.reverse_lookup(factory) if factory else None

    def get_pass_manager(self):
        kwargs = self.pass_manager_kwargs
        if kwargs is None:
            kwargs = {}
        kwargs["backend"] = self.backend
        return self.pass_manager_factory(**kwargs)
    
    def get_synthesizer(self):
        return SynthesizerRegistry.from_dict(self.synthesizer_config)
    
    def get_compiler_steps(self):
        if self.compiler_config.get("steps") is not None:
            return [StepRegistry.from_dict(s) for s in self.compiler_config["steps"]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name" : self.name,
            "synthesizer_config": self.synthesizer_config,
            "compiler_config" : self.compiler_config,
            "pass_manager_factory_name" : self.pass_manager_factory_name,
            "pass_manager_kwargs" : self.pass_manager_kwargs,
            "backend_name" : self.backend_name
        }
    @classmethod
    def from_dict(cls, data) -> "PipelineConfig":
        return PipelineConfig(
            name=data.get("name"),
            synthesizer_config=data.get("synthesizer_config"),
            compiler_config=data.get("compiler_config"),
            pass_manager_factory_name=data.get("pass_manager_factory_name"),
            pass_manager_kwargs=data.get("pass_manager_kwargs"),
            backend_name=data.get("backend_name")
        )

    def __eq__(self, other):
        if not isinstance(other, PipelineConfig):
            return False
        return (
            self.name == other.name and
            self.synthesizer_config == other.synthesizer_config and
            self.compiler_config == other.compiler_config and
            self.pass_manager_factory_name == other.pass_manager_factory_name and
            self.pass_manager_kwargs == other.pass_manager_kwargs and
            self.backend_name == other.backend_name
        )



