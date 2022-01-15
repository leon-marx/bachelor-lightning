from pytorch_lightning.callbacks import Callback


class ImageLogger(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        pass

    #####################################################
    # 1) DO THIS with 0. element of each domain/content folder
    # -> for grad plot: think of something
    # 2) lr scheduler