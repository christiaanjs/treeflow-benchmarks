import numpy as np
from treeflow_benchmarks.bito import BeagleLikelihoodBenchmarkable


class BeagleDirectLikelihoodBenchmarkable(BeagleLikelihoodBenchmarkable):
    def initialize(
        self, topology_file, fasta_file, model, calculate_clock_rate_gradients
    ):
        self.calculate_clock_rate_gradients = calculate_clock_rate_gradients
        super().initialize(
            topology_file, fasta_file, model, calculate_clock_rate_gradients
        )
        self.branch_length_state = np.array(
            self.inst.tree_collection.trees[0].branch_lengths, copy=False
        )

    def calculate_likelihoods(self, branch_lengths: np.ndarray, params) -> np.ndarray:
        self.branch_length_state[:-1] = branch_lengths
        res = np.array(self.inst.log_likelihoods())[0]
        # output[i] = res
        return res

    def extract_substitution_model_grads(self, bito_gradient, subst_grad_dict):
        if self.phylo_model.subst_model == "gtr":  # TODO: Fix
            subst_grad_dict["rates"] = np.array(bito_gradient["substitution_model"])[
                :-2
            ]
        elif self.phylo_model.subst_model == "hky":
            subst_grad_dict["kappa"] = np.array(bito_gradient["substitution_model"])[0]
        if self.phylo_model.subst_model != "jc":
            subst_grad_dict["frequencies"] = np.array(
                bito_gradient["substitution_model"]
            )[-4:]

    def calculate_gradients(self, branch_lengths: np.ndarray, params):
        self.branch_length_state[:-1] = branch_lengths
        gradient = self.inst.phylo_gradients()[0]
        branch_gradient_array = np.array(gradient.gradient["branch_lengths"])
        param_gradient = dict(
            clock_model_params=dict(),
            subst_model_params=dict(),
            site_model_params=dict(),
        )
        if (
            self.calculate_clock_rate_gradients
            and "clock_rate" in params["clock_model_params"]
        ):
            param_gradient["clock_model_params"]["clock_rate"] = np.array(
                gradient.gradient["clock_model"]
            )
        if "site_weibull_concentration" in params["site_model_params"]:
            param_gradient["site_model_params"][
                "site_weibull_concentration"
            ] = np.array(gradient.gradient["site_model"])
        self.extract_substitution_model_grads(
            gradient.gradient, param_gradient["subst_model_params"]
        )
        output = [branch_gradient_array[:-1], param_gradient]
        return output
