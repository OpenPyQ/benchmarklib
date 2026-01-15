
'''
import json
from typing import Dict, List, Any, Optional, Type, Callable
from datetime import datetime
import inspect

from qiskit.transpiler import PassManager
import qiskit.transpiler.passes as passes_module
from qiskit.transpiler.basepasses import BasePass

# import additional registries
from .pipeline_steps import StepRegistry
from .synthesis.synthesizer import SynthesizerRegistry

class PassRegistry:
    """
    Registry for mapping pass names to pass classes.
    Supports both built-in Qiskit passes and custom user-defined passes.
    """

    _registry: Dict[str, Type[BasePass]] = {}
    
    @classmethod
    def _register_builtin_passes(cls):
        """Automatically register base Qiskit passes"""
        
        builtin_passes = {}

        for name in dir(passes_module):
            item = getattr(passes_module, name)
            if isinstance(item, type) and issubclass(item, BasePass):
                builtin_passes[item.__name__] = item

        # add additional passes if installed
        try:
            from qiskit_ibm_transpiler.ai.routing import AIRouting
            builtin_passes['AIRouting'] = AIRouting
        except ImportError:
            print("qiskit-ibm-transpiler[ai-local-mode] not installed; skipping AIRouting pass registration")
            pass
        
        cls._registry.update(builtin_passes)
    
    @classmethod
    def register(cls, pass_class: Type[BasePass]):
        """
        Register a custom pass.
        Can be used as a decorator or with function call.
        Examples:
            @PassRegistry.register
            class MyPass(BasePass):
                ...

            class OtherPass(TransformationPass):
                ... 
            PassRegistry.register(OtherPass)
        
        """
        if not inspect.isclass(pass_class) or not issubclass(pass_class, BasePass):
            raise ValueError(f"{pass_class} must be a class inheriting from BasePass")
        
        cls._registry[pass_class.__name__] = pass_class
        return pass_class
    
    @classmethod
    def get(cls, name: str) -> Type[BasePass]:
        """
        Get a pass class by name.
        
        Args:
            name: Name of the pass
            
        Returns:
            The pass class
            
        Raises:
            KeyError: If pass name is not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Pass '{name}' not found in registry. Available passes: {list(cls._registry.keys())}")
        return cls._registry[name]
    
# initialize registry with Qiskit builtin passes
PassRegistry._register_builtin_passes()
    

class ConditionRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Condition function '{name}' not found in registry.")
        return cls._registry[name]

    @classmethod
    def reverse_lookup(cls, func):
        """Find the name of a registered function."""
        for name, f in cls._registry.items():
            if f == func:
                return name
        # Fallback for named functions not explicitly registered via decorator
        if hasattr(func, '__name__') and func.__name__ in cls._registry:
             return func.__name__
        raise ValueError(f"Condition function {func} is not registered. Cannot serialize.")    
    
    @classmethod
    def _register_builtin_conditions(cls):
        """Register some common condition functions."""
        
        from qiskit.passmanager import flow_controllers
        pass

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable, Union

from sqlalchemy import Boolean, Column, ForeignKey, Index, Integer, String, JSON, Float, DateTime, LargeBinary, select, create_engine, select, func, or_, text
from sqlalchemy.orm import  declared_attr, Mapped, relationship, mapped_column, sessionmaker, Session, DeclarativeBase, DeclarativeMeta, Mapped, relationship, selectinload, joinedload
from sqlalchemy.ext.mutable import MutableDict

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import *
from qiskit.passmanager.flow_controllers import FlowControllerLinear, DoWhileController, ConditionalController, BaseController


from benchmarklib.core.base import Base
from .registries import PassRegistry, SynthesizerRegistry, StepRegistry, ConditionRegistry


class PipelineConfig(Base):
    __tablename__ = "pipeline_configs"

    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    synthesizer_config: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)
    compiler_config: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)
    pass_manager_config: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)
    backend_name: Mapped[Optional[str]]

    def __init__(self, *args, **kwargs):
        if "pass_manager" in kwargs:
            kwargs["pass_manager_config"] = create_config_from_pass_manager(kwargs.pop("pass_manager"))

        super().__init__(*args, **kwargs)


    def get_pass_manager(self):
        return create_pass_manager_from_config(self.pass_manager_config)
    
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
            "pass_manager_config" : self.pass_manager_config,
            "backend_name" : self.backend_name
        }
    @classmethod
    def from_dict(cls, data) -> "PipelineConfig":
        return PipelineConfig(
            name=data.get("name"),
            synthesizer_config=data.get("synthesizer_config"),
            compiler_config=data.get("compiler_config"),
            pass_manager_config=data.get("pass_manager_config"),
            backend_name=data.get("backend_name")
        )

def create_pass_manager_from_config(config: Dict[str, Any]) -> PassManager:
    """Reconstructs a PassManager from the serialized config."""
    if config.get("type") == "empty":
        return PassManager()

    # The recursive builder returns a FlowController or a Pass
    # PassManager requires a list of passes/controllers in append()
    reconstructed_flow = _reconstruct_task(config)
    
    pm = PassManager()
    
    # If the root is a Linear controller, we append its children. 
    # If it is a Pass or a complex Controller, we append it directly.
    if isinstance(reconstructed_flow, FlowControllerLinear) and not isinstance(reconstructed_flow, (DoWhileController, ConditionalController)):
        for task in reconstructed_flow.tasks:
            pm.append(task)
    else:
        pm.append(reconstructed_flow)
        
    return pm

def serialize_pass_manager(pm: PassManager) -> Dict[str, Any]:
    if pm is None:
        return {"type": "empty"}
    
    flow_controller = pm.to_flow_controller()
    return serialize_task(flow_controller)

def serialize_task(task: Union[BasePass, BaseController]) -> Dict[str, Any]:
    """
    Recursively serialize a Pass or FlowController
    """
    # handle leaf node (BasePass)
    if isinstance(task, BasePass):
        return _serialize_base_pass(task)
    
    serialized = {
        "tasks": [serialize_task(t) for t in task.tasks]
    }

    if isinstance(task, DoWhileController):
        serialized["type"] = "do_while"
        serialized["condition"] = ConditionRegistry.reverse_lookup(task.do_while)

    elif isinstance(task, ConditionalController):
        serialized["type"] = "conditional"
        serialized["condition"] = ConditionRegistry.reverse_lookup(task.condition)

    elif isinstance(task, FlowControllerLinear):
        serialized["type"] = "linear"

    else:
        raise TypeError(f"Unsupported task type: {type(task)}")
    
    return serialized

def _serialize_base_pass(p: BasePass) -> Dict[str, Any]:
    cls_name = p.__class__.__name__
    PassRegistry.get(cls_name)  # throws ValueError if name is not registerred
    # exclude known temporary attributes we don't want to serialize
    exclude = {'property_set', 'target', '_hash'}
    params = {}
    for key, value in p.__dict__.items():
        if key not in exclude and not key.startswith("_"):
            params[key] = value

    return {
        "type": "pass",
        "name": cls_name,
        "params": params,
    }

def _reconstruct_task(config: Dict[str, Any]):
    task_type = config.get("type")

    if task_type == "pass":
        pass_cls = PassRegistry.get(config["name"])
        # Unpack params into constructor
        return pass_cls(**config.get("params", {}))

    # If it's a controller, we first reconstruct its children
    children = [_reconstruct_task(t) for t in config.get("tasks", [])]

    if task_type == "linear":
        return FlowControllerLinear(children)

    if task_type == "do_while":
        condition_func = ConditionRegistry.get(config["condition"])
        return DoWhileController(children, do_while=condition_func)

    if task_type == "conditional":
        condition_func = ConditionRegistry.get(config["condition"])
        return ConditionalController(children, condition=condition_func)

    raise ValueError(f"Unknown task type in config: {task_type}")

def serialize_pass(p: BasePass) -> Dict[str, Any]:
    """
    Convert a pass instance into metadata describing how to rebuild it.
    """

    cls = p.__class__
    name = cls.__name__

    PassRegistry.get(name)  # throws ValueError if name is not registerred

    # Extract constructor parameters by reading __dict__
    params = {
        key: value
        for key, value in p.__dict__.items()
        if not key.startswith("_")
    }

    return {
        "name": name,
        "params": params,
    }

def create_config_from_pass_manager(pass_manager: Optional[PassManager]) -> Dict[str, Any]:
    return serialize_pass_manager(pass_manager)
'''