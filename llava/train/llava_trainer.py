from transformers import Trainer


class LLaVATrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir=None, state_dict=None):
        super(LLaVATrainer, self)._save(output_dir, state_dict)
