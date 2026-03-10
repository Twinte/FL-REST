from .fedavg import FedAvg
from .fedopt import FedOpt
from .scaffold import ScaffoldStrategy
from .fedprune import FedPruneStrategy
from .heterofl import HeteroFLStrategy
from .fedrolex import FedRolexStrategy
from .fiarse import FIARSEStrategy
from .fluid import FLuIDStrategy
import torch.optim as optim

def get_strategy(strategy_name, global_model=None, lr=1.0, momentum=0.0, **kwargs):
    """
    Factory to return the requested strategy.
    
    Partial-training strategies (FedPrune, HeteroFL, FedRolex, FIARSE, FLuID)
    share the same base class and client-side gradient masking.
    """
    if strategy_name == "FedAvg":
        return FedAvg()
    
    elif strategy_name == "Scaffold":
        if global_model is None:
            raise ValueError("Scaffold requires global_model")
        return ScaffoldStrategy(global_model)
    
    elif strategy_name == "FedAvgM":
        if global_model is None:
            raise ValueError("FedAvgM requires global_model instance")
        return FedOpt(
            global_model, 
            optimizer_cls=optim.SGD, 
            lr=lr, 
            momentum=momentum
        )
        
    elif strategy_name == "FedAdam":
        if global_model is None:
            raise ValueError("FedAdam requires global_model instance")
        return FedOpt(
            global_model,
            optimizer_cls=optim.Adam,
            lr=lr
        )
    
    elif strategy_name == "FedPrune":
        if global_model is None:
            raise ValueError("FedPrune requires global_model instance")
        return FedPruneStrategy(
            global_model,
            ema_decay=kwargs.get("ema_decay", 0.9),
            importance_alpha=kwargs.get("importance_alpha", 0.3),
            total_rounds=kwargs.get("total_rounds", 100),
            ramp_range=kwargs.get("ramp_range", 0.3),
        )
    
    elif strategy_name == "HeteroFL":
        if global_model is None:
            raise ValueError("HeteroFL requires global_model instance")
        return HeteroFLStrategy(
            global_model,
            total_rounds=kwargs.get("total_rounds", 100),
        )
    
    elif strategy_name == "FedRolex":
        if global_model is None:
            raise ValueError("FedRolex requires global_model instance")
        return FedRolexStrategy(
            global_model,
            total_rounds=kwargs.get("total_rounds", 100),
        )
    
    elif strategy_name == "FIARSE":
        if global_model is None:
            raise ValueError("FIARSE requires global_model instance")
        return FIARSEStrategy(
            global_model,
            ema_decay=kwargs.get("ema_decay", 0.9),
            total_rounds=kwargs.get("total_rounds", 100),
        )
    
    elif strategy_name == "FLuID":
        if global_model is None:
            raise ValueError("FLuID requires global_model instance")
        return FLuIDStrategy(
            global_model,
            leader_threshold=kwargs.get("leader_threshold", 0.7),
            ema_decay=kwargs.get("ema_decay", 0.9),
            total_rounds=kwargs.get("total_rounds", 100),
        )
        
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")