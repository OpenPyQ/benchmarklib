import json
from typing import Dict, List, Any, Optional, Type, Callable
from datetime import datetime
import inspect

from qiskit import generate_preset_pass_manager
from qiskit_ibm_transpiler import generate_ai_pass_manager
from qiskit.transpiler import PassManager

# import additional registries
from .pipeline_steps import StepRegistry
from .synthesis.synthesizer import SynthesizerRegistry

class PassManagerFactoryRegistry:
    """
    Registry for mapping names to pass manager factories.
    A factory is a callable that should accept JSON serializable keyword arguments (with exception of backend) and returns a PassManager.
    The factory will be passed any stored kwargs along with backend=backend when invoked.
    """

    _registry: Dict[str, Callable[..., PassManager]] = {}
    
    @classmethod
    def register(cls, factory: Callable[..., PassManager]):
        """
        Register a custom pass.
        Can be used as a decorator or with function call.
        Examples:
            @PassManagerFactoryRegistry.register
            def my_pass_factory(**kwargs) -> PassManager:
                ...

            def other_pass_factory(seed, optimization_level, **kwargs) -> PassManager:
                ... 
            PassManagerFactoryRegistry.register(other_pass_factory)
        
        """
        if not callable(factory):
            raise ValueError(f"{factory} must be a callable that returns a PassManager")
        
        cls._registry[factory.__name__] = factory
        return factory
    
    @classmethod
    def get(cls, name: str) -> Callable[..., PassManager]:
        """
        Get a pass manager factory by name.
        
        Args:
            name: Name of the pass manager factory
            
        Returns:
            The pass manager factory
            
        Raises:
            KeyError: If pass manager factory is not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Pass manager factory '{name}' not found in registry. Available factories: {list(cls._registry.keys())}")
        return cls._registry[name]
    
    @classmethod
    def reverse_lookup(cls, factory: Callable[..., PassManager]) -> str:
        """
        Get the name of a registered pass manager factory.
        
        Args:
            factory: The pass manager factory callable
            
        Returns:
            The name of the pass manager factory
            
        Raises:
            ValueError: If the factory is not registered
        """
        for name, reg_factory in cls._registry.items():
            if reg_factory == factory:
                return name
        raise ValueError("Pass manager factory not found in registry.")
    
# initialize registry with Qiskit builtin
PassManagerFactoryRegistry.register(generate_preset_pass_manager)
PassManagerFactoryRegistry.register(generate_ai_pass_manager)
    

