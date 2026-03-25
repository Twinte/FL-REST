from .standard import StandardSGD
from .fedprox import FedProx
from .scaffold import Scaffold
from .partial_training_client import PartialTrainingClient
from .fiarse_client import FIARSEClient
import config

def get_client_algorithm(algo_name):
    if algo_name == "Standard":
        return StandardSGD()
    elif algo_name == "FedProx":
        return FedProx(mu=config.FEDPROX_MU)
    elif algo_name == "Scaffold":
        return Scaffold()

    # All partial-training methods share the same client algorithm.
    # The server-side strategy determines WHICH neurons each client trains;
    # the client just receives indices and masks gradients.
    elif algo_name == "FedPrune":
        return PartialTrainingClient(method_name="FedPrune")
    elif algo_name == "HeteroFL":
        return PartialTrainingClient(method_name="HeteroFL")
    elif algo_name == "FedRolex":
        return PartialTrainingClient(method_name="FedRolex")

    # FIARSE (CORRECTED): TCB-GD training with biased gradients.
    # No separate importance computation, no importance uploads.
    # importance_batches accepted for backward compat but unused.
    elif algo_name == "FIARSE":
        return FIARSEClient(importance_batches=5)

    # FLuID: non-stragglers and stragglers both use gradient masking.
    # Non-stragglers receive full neuron indices (no actual masking),
    # stragglers receive submodel indices (invariant neurons dropped).
    elif algo_name == "FLuID":
        return PartialTrainingClient(method_name="FLuID")

    else:
        raise ValueError(f"Unknown Client Algorithm: {algo_name}")
