import numpy as np
from rdkit import Chem


class Reagent:
    __slots__ = [
        "reagent_name",
        "smiles",
        "num_props",
        "min_uncertainty",
        "initial_scores",
        "mol",
        "known_var",
        "current_mean",
        "current_std",
        "current_phase"
    ]

    def __init__(self, reagent_name: str, smiles: str, num_props: int = 1):
        """
        Basic init
        :param reagent_name: Reagent name
        :param smiles: smiles string
        :param num_props: number of properties being optimized
        """
        self.smiles = smiles
        self.reagent_name = reagent_name
        self.num_props = num_props
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.initial_scores = []
        self.known_var = None  # Will be initialized during init_given_prior
        self.current_phase = "warmup"
        self.current_mean = np.zeros(num_props)
        self.current_std = np.zeros(num_props)

    def add_score(self, score: list[float]) -> None:
        """
        Either adds a score to self.initial_scores if self._current_phase == "warmup", otherwise, does the bayesian
        update of the mean and standard deviation.
        :param score: New score collected for the reagent
        :return: None
        """
        if self.current_phase == "search":
            current_var = self.current_std ** 2
            # Then do the bayesian update
            self.current_mean = self._update_mean(current_var=current_var, observed_value=score)
            self.current_std = self._update_std(current_var=current_var)
        elif self.current_phase == "warmup":
            self.initial_scores.append(score)
        else:
            raise ValueError(f"self.current_phase should be warmup or search, found {self.current_phase}")
        return

    def sample(self) -> list[float]:
        """
        Takes a random sample from the prior distribution
        :return: sample from the prior distribution
        """
        if self.current_phase != "search":
            raise ValueError(f"Must call Reagent.init() before sampling")
        return [np.random.normal(loc=self.current_mean[i], scale=self.current_std[i]) for i in range(self.num_props)]

    def init_given_prior(self, prior_mean: list[float], prior_std: list[float]) -> None:
        """
        After warmup, set the prior distribution from the given parameters and replay the warmup scores.

        The meaning of "prior" here is the distribution before any scores have been seen for this reagent.
        This would typically be the from the score distribution seen across all reagents during the warm up phase.
        The specific values seen during warmup (stored in initial_scores) will then be run as updates just
        as they would be during the regular search phase.

        :param prior_mean: Mean of the prior distribution
        :param prior_std: Standard deviation of the prior distribution
        """
        if self.current_phase != "warmup":
            raise ValueError(f"Reagent {self.reagent_name} has already been initialized.")
        elif not self.initial_scores:
            raise ValueError(f"Must collect initial scores before initializing Reagent {self.reagent_name}")

        self.current_std = prior_std
        self.current_mean = prior_mean
        # This is an interesting assumption. Namely that the standard deviation of the
        # distribution of a reagent is estimated by the standard deviation across all reagents
        # during warmup.
        # Likely, each reagent has a smaller standard deviation than the one across all warmup
        # but this still practically works well.
        self.known_var = prior_std ** 2

        self.current_phase = "search"

        for score in self.initial_scores:
            self.add_score(score)

    def _update_mean(self, current_var: list[float], observed_value: list[float]) -> list[float]:
        """
        Bayesian update to the mean
        :param current_var: The current variance
        :param observed_value: value to use to update the mean
        :return: the updated mean
        """
        numerator = current_var * observed_value + self.known_var * self.current_mean
        denominator = current_var + self.known_var
        return numerator / denominator

    def _update_std(self, current_var: list[float]) -> list[float]:
        """
        Bayesian update to the standard deviation
        :param current_var: The current variance
        :return: the updated standard deviation
        """
        numerator = current_var * self.known_var
        denominator = current_var + self.known_var
        return np.sqrt(numerator / denominator)
