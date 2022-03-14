import atexit
import os
import tarfile
import tempfile

from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.models.archival import _cleanup_archive_dir
from allennlp.models.model import Model

from core.utils.logging import logger
from core.pretrain.scripts.util import cached_path
try:
    from apex import amp
except ImportError:
    amp = None

_DEFAULT_WEIGHTS = "best.th"
CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "weights.th"


def get_model_and_dataset_reader(args,
                                 overrides: str = "",):
    config, serialization_dir = get_config(args.archive_file, overrides)
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    model = load_model_from_path(config,
                                 serialization_dir,
                                 weights_file=args.weights_file,
                                 cuda_device=args.cuda_device)
    return dataset_reader, model


def get_config(archive_file: str, overrides: str = ""):
    # redirect to the cache, if necessary
    resolved_archive_file = cached_path(archive_file)

    if resolved_archive_file == archive_file:
        logger.info(f"loading archive file {archive_file}")
    else:
        logger.info(f"loading archive file {archive_file} from cache at {resolved_archive_file}")

    if os.path.isdir(resolved_archive_file):
        serialization_dir = resolved_archive_file
    else:
        # Extract archive to temp dir
        tempdir = tempfile.mkdtemp()
        logger.info(f"extracting archive file {resolved_archive_file} to temp dir {tempdir}")
        with tarfile.open(resolved_archive_file, "r:gz") as archive:
            archive.extractall(tempdir)
        # Postpone cleanup until exit in case the unarchived contents are needed outside
        # this function.
        atexit.register(_cleanup_archive_dir, tempdir)

        serialization_dir = tempdir
    # Load config
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), overrides)
    return config, serialization_dir


def load_model_from_path(config,
                         serialization_dir,
                         cuda_device: int = -1,
                         opt_level: str = None,
                         weights_file: str = None, ) -> Model:
    """
       Instantiates an Archive from an archived `tar.gz` file.

       # Parameters

       archive_file : `str`
           The archive file to load the model from.
       cuda_device : `int`, optional (default = `-1`)
           If `cuda_device` is >= 0, the model will be loaded onto the
           corresponding GPU. Otherwise it will be loaded onto the CPU.
       opt_level : `str`, optional, (default = `None`)
           Each `opt_level` establishes a set of properties that govern Amp’s implementation of pure or mixed
           precision training. Must be a choice of `"O0"`, `"O1"`, `"O2"`, or `"O3"`.
           See the Apex [documentation](https://nvidia.github.io/apex/amp.html#opt-levels-and-properties) for
           more details. If `None`, defaults to the `opt_level` found in the model params. If `cuda_device==-1`,
           Amp is not used and this argument is ignored.
       overrides : `str`, optional (default = `""`)
           JSON overrides to apply to the unarchived `Params` object.
       weights_file : `str`, optional (default = `None`)
           The weights file to use.  If unspecified, weights.th in the archive_file will be used.
       """

    if weights_file:
        weights_path = weights_file
    else:
        weights_path = os.path.join(serialization_dir, _WEIGHTS_NAME)
        # Fallback for serialization directories.
        if not os.path.exists(weights_path):
            weights_path = os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

    model = Model.load(
        config.duplicate(),
        weights_file=weights_path,
        serialization_dir=serialization_dir,
        cuda_device=cuda_device,
        opt_level=opt_level,
    )

    return model
