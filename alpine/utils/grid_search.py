
import itertools
from alpine.trainers.alpine_base import AlpineBaseModule
import pandas as pd
from tqdm import tqdm
class GridSearch:
    def __init__(
        self,
        model_class,
        model_params,
        compile_params,
        n_iters=[2000],
        device="cpu",
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
    
    def _generate_combos(self, grid):
        """Internal helper to create cartesian products."""
        if not grid:
            return [{}]
        keys, values = zip(*grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    
    def run(self, coords, signal, metric_trackers=None, verbose=True):
        self.results = []
        model_param_combos = self._generate_combos(self.model_params)
        compile_param_combos = self._generate_combos(self.compile_params)
        
        num_iterations = len(model_param_combos) * len(compile_param_combos) * len(self.n_iters)
        if verbose:
            print(f"Starting grid search with {num_iterations} candidates on {self.device}.")
            outer_pbar = tqdm(total=num_iterations, desc="Grid Search Progress", position=0)
    
        for model_params in model_param_combos:
            for compile_params in compile_param_combos:
                for n_iters_param in self.n_iters:
                    try:
                        model = self.model_class(**model_params).to(self.device)
                        model.compile(**compile_params)
                        if verbose:
                            config_str = f"Arch: {model_params} | Optim: {compile_params} | Iters: {n_iters_param}"
                            outer_pbar.write(f"Running: {config_str}")
                            
                        kwargs = {}
                        if verbose:
                            kwargs = {
                                'tqdm_kwargs': {'position': 1, 'leave': False, 'desc': 'Training'}
                            }
                        result = model.fit_signal(
                            input = coords,
                            signal = signal,
                            enable_tqdm = verbose,
                            return_features = False,
                            track_loss_history = False,
                            n_iters=n_iters_param,
                            metric_trackers=metric_trackers,
                            kwargs=kwargs,
                        )
                        
                        final_loss = result.get('loss', float('nan'))
                        metrics = result.get('metrics', {})
                        status = "Success"
                    except Exception as e:
                        if verbose:
                            outer_pbar.write(f"--> FAILED: {e}")
                        final_loss = float('nan')
                        metrics = {}
                        status = f"Failed: {e}"
                        
                    if verbose:
                        outer_pbar.set_postfix({'Last Loss': f"{final_loss:.6f}"})
                        outer_pbar.update(1)
                    
                    record = {
                        'status': status,
                        'loss': final_loss,
                        **metrics,
                        **model_params,
                        **compile_params,
                        'n_iters': n_iters_param,
                    }
                    self.results.append(record)
                    
        if verbose:
            outer_pbar.close()
        return self.results
    
    def get_results_df(self):
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results).sort_values(by='loss')
                    
        
        
    
    
    