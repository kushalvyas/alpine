
import itertools
from alpine.trainers.alpine_base import AlpineBaseModule

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
    
        for model_params in model_param_combos:
            for compile_params in compile_param_combos:
                for n_iters_param in self.n_iters:
                    model = self.model_class(**model_params).to(self.device)
                    model.compile(**compile_params)
                    if verbose:
                       config_str = f"Arch: {model_params} | Optim: {compile_params} | Iters: {n_iters_param}"
                       print(f"Running configuration: {config_str}")
                    result = model.fit_signal(
                        input = coords,
                        signal = signal,
                        enable_tqdm = verbose,
                        return_features = False,
                        track_loss_history = False,
                        n_iters=n_iters_param,
                        metric_trackers=metric_trackers,
                    )
                    
                    final_loss = result['loss']
                    if verbose:
                        print(f"--> Loss: {final_loss:.6f}")
                        
                    self.results.append({
                        'model_params': model_params,
                        'compile_params': compile_params,
                        'n_iters_param': n_iters_param,
                        'result': result,
                    })
                    
        return self.results
                    
        
        
    
    
    