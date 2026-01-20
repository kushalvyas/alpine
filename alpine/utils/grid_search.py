import gc
import itertools
import os

import pandas as pd
import torch
from tqdm.auto import tqdm

from alpine.trainers.alpine_base import AlpineBaseModule


class GridSearch:
    def __init__(
        self,
        model_class,
        model_params,
        compile_params,
        n_iters=[2000],
        device="cpu",
        save_path="grid_search_results.csv",
    ):
        assert isinstance(model_params, dict), "model_params must be a dictionary"
        assert isinstance(compile_params, dict), "compile_params must be a dictionary"
        assert isinstance(n_iters, list), "n_iters must be a list"
        assert issubclass(model_class, AlpineBaseModule)

        self.model_class = model_class
        self.model_params = model_params
        self.compile_params = compile_params
        self.n_iters = n_iters
        self.device = device
        self.save_path = save_path

        # Track best results
        self.best_loss = float("inf")
        self.best_state_dict = None
        self.best_config = None

    def _generate_combos(self, grid):
        if not grid:
            return [{}]
        keys, values = zip(*grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _get_metrics(self, val):
        if torch.is_tensor(val):
            val = val.detach().cpu()
            if val.numel() == 1:
                return val.item()
            val = val.tolist()

        if isinstance(val, list) and len(val) > 0:
            return val[-1]

        return val

    def run(self, coords, signal, metric_trackers=None, verbose=True):
        self.results = []
        model_combos = self._generate_combos(self.model_params)
        compile_combos = self._generate_combos(self.compile_params)

        total_runs = len(model_combos) * len(compile_combos) * len(self.n_iters)

        if verbose:
            print(
                f"Starting {total_runs} runs on {self.device}. Saving to {self.save_path}"
            )
            outer_pbar = tqdm(total=total_runs, desc="Grid Search", position=0)

        for m_params in model_combos:
            for c_params in compile_combos:
                for n_iters in self.n_iters:
                    try:
                        # Instantiate
                        model = self.model_class(**m_params).to(self.device)
                        model.compile(**c_params)

                        if verbose:
                            config_str = (
                                f"Arch:{m_params} | Opt:{c_params} | Iters:{n_iters}"
                            )
                            outer_pbar.write(f"Running: {config_str}")

                        # Configure Inner Bar
                        kwargs = {}
                        if verbose:
                            kwargs = {
                                "tqdm_kwargs": {
                                    "position": 1,
                                    "leave": False,
                                    "desc": "Training",
                                }
                            }

                        # Train
                        result = model.fit_signal(
                            input=coords,
                            signal=signal,
                            enable_tqdm=verbose,
                            return_features=False,
                            track_loss_history=False,
                            n_iters=n_iters,
                            metric_trackers=metric_trackers,
                            kwargs=kwargs,
                        )

                        # Extract & Sanitize Data
                        final_loss = result.get("loss", float("nan"))
                        if torch.is_tensor(final_loss):
                            final_loss = final_loss.item()

                        raw_metrics = result.get("metrics", {})
                        clean_metrics = {
                            k: self._get_metrics(v) for k, v in raw_metrics.items()
                        }
                        status = "Success"

                        # Check for Winner (Save weights to CPU)
                        if final_loss < self.best_loss:
                            self.best_loss = final_loss
                            self.best_config = {
                                **m_params,
                                **c_params,
                                "n_iters": n_iters,
                            }
                            # Deepcopy to CPU so it persists after 'model' is deleted
                            self.best_state_dict = {
                                k: v.cpu().clone()
                                for k, v in model.state_dict().items()
                            }

                    except Exception as e:
                        if verbose:
                            outer_pbar.write(f"--> FAILED: {e}")
                        final_loss = float("inf")
                        clean_metrics = {}
                        status = f"Failed: {str(e)}"

                    # Record Keeping
                    record = {
                        "status": status,
                        "loss": final_loss,
                        **clean_metrics,
                        **m_params,
                        **c_params,
                        "n_iters": n_iters,
                    }
                    self.results.append(record)

                    # Auto-Save to CSV (Incremental)
                    df_row = pd.DataFrame([record])
                    # Write header only if file doesn't exist
                    header = not os.path.exists(self.save_path)
                    df_row.to_csv(self.save_path, mode="a", header=header, index=False)

                    # Visual Update
                    if verbose:
                        outer_pbar.set_postfix(
                            {
                                "Best": f"{self.best_loss:.6f}",
                                "Curr": f"{final_loss:.6f}",
                            }
                        )
                        outer_pbar.update(1)

                    # Cleanup
                    del model
                    if self.device != "cpu":
                        torch.cuda.empty_cache()
                    gc.collect()

        if verbose:
            outer_pbar.close()
        return self.results

    def get_results_df(self):
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results).sort_values(by="loss")

    def get_best_model(self):
        """Recreates the winning model with trained weights."""
        if self.best_state_dict is None:
            print("No successful runs yet.")
            return None

        # Extract args from best_config
        m_args = {k: v for k, v in self.best_config.items() if k in self.model_params}
        c_args = {k: v for k, v in self.best_config.items() if k in self.compile_params}

        model = self.model_class(**m_args)
        model.compile(**c_args)
        model.load_state_dict(self.best_state_dict)
        return model
