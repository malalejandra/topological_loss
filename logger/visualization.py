import importlib
from datetime import datetime

# from pathlib import Path


class TensorboardWriter:
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = (
                    "Warning: visualization (Tensorboard) is configured to use, but currently not installed on "
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to "
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                )
                logger.warning(message)

        self.step = 0
        self.mode = ""

        self.tb_writer_ftns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
            "add_custom_scalars",
        }
        self.tag_mode_exceptions = {
            "add_histogram",
            "add_text",
            "add_embedding",
            "add_custom_scalars",
        }
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        #print("vis:",name)
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)
            #print("add_data:",add_data)
            if name not in self.tag_mode_exceptions:

                def wrapper(tag, data, *args, **kwargs):
                    if add_data is not None:
                        # add mode(train/valid) tag

                        tag = f"{tag}/{self.mode}"
                        #print("tag",name,tag)
                        add_data(tag, data, self.step, *args, **kwargs)

                return wrapper
            else:
                return add_data
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    f"type object '{self.selected_module}' has no attribute '{name}'"
                )
            return attr
