import click
import glob
import numpy as np
import os
import pathlib
import tifffile as tiff
from rich import traceback, print
from model.unet_instance import *
from utils import weights_init
from torchvision.datasets.utils import download_url
from urllib.error import URLError

from captum.attr import GuidedGradCam

import torch.nn as nn

WD = os.path.dirname(__file__)


@click.command()
@click.option('-i', '--input', required=True, type=str, help='Path to data file to predict.')
@click.option('-m', '--model', type=str, default="U-Net",
              help='Path to an already trained XGBoost model. If not passed a default model will be loaded.')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-s/-ns', '--sanitize/--no-sanitize', type=bool, default=False,
              help='Whether to remove model after prediction or not.')
@click.option('-suf', '--suffix', type=str, help='Suffix for output files, eg: ".tif" or ".npy"')
@click.option('-o', '--output', default="", required=True, type=str, help='Path to write the output to')
@click.option('-f', '--feat', default='_feat.ome.tif', type=str, help='Filename for ggcam features output')
@click.option('-t', '--target', required=True, type=int,
              help='Output indices for which gradients are computed (target class)')
@click.option('-h', '--ome', type=bool, default=True,
              help='human readable output (OME-TIFF format), input and output as image channels')
@click.option('--architecture', type=str, default="U-Net", help="U-Net or CU-Net")

def main(input: str, suffix: str, model: str, cuda: bool, output: str, sanitize: bool, feat: str, target: int,
         ome: bool, architecture: str):
    """Command-line interface for rts-feat-imp"""

    print(r"""[bold blue]
        rts-feat-imp
        """)

    print('[bold blue]Run [green]rts-feat-imp --help [blue]for an overview of all commands\n')

    out_filename = feat
    target_class = target

    print('[bold blue] Calculating Guided Grad-CAM features...')
    print('[bold blue] Target class: ' + str(target_class))

    model = get_pytorch_model(model, False, architecture)
    if cuda:
        model.cuda()

    print('[bold blue] Parsing data...')
    if os.path.isdir(input):
        input_list = glob.glob(os.path.join(input, "*"))
        for inputs in input_list:
            print(f'[bold yellow] Input: {inputs}')
            file_feature_importance(inputs, model, target_class, inputs.replace(input, output).replace(".tif", suffix),
                                    ome_out=ome)
    else:
        file_feature_importance(input, model, target_class, output, ome_out=ome)

    if sanitize:
        os.remove(os.path.join(f'{WD}', "models", "models/U_NET.ckpt"))


def file_feature_importance(input, model, target_class, output, ome_out=False):
    input_data, label_data = read_data_to_predict(input)

    feat_ggcam = features_ggcam(model, input_data, target_class)

    if ome_out:
        print(f'[bold green] Output: {output}_ggcam_t_{target_class}.ome.tif')
        write_ome_out(input_data, feat_ggcam, output + "_ggcam_t_" + str(target_class))
    else:
        print(f'[bold green] Output: {output}_ggcam_t_{target_class}.npy')
        write_results(feat_ggcam, output + "_ggcam_t_" + str(target_class))

    # print(f'[bold green] Output: {output}_ggcam_t_{target_class}')

    #write_ome_out(input_data, feat_ggcam, output + "_ggcam_t_" + str(target_class))
    write_results(feat_ggcam, output + "_ggcam_t_" + str(target_class))


def features_ggcam(net, data_to_predict, target_class):
    """
    features ggcam
    GuidedGradCam implementation for all features
    """

    net.eval()

    img = data_to_predict
    img = torch.from_numpy(np.expand_dims(img, 0)).float()

    wrapped_net = agg_segmentation_wrapper_module(net)

    guided_gc = GuidedGradCam(wrapped_net, wrapped_net._model.final)

    gc_attr = guided_gc.attribute(inputs = img, target=target_class)

    gc_attr = torch.abs(gc_attr)

    #print("ggcam out shape: " + str(gc_attr.shape))
    img_out = gc_attr.squeeze(0).cpu().detach().numpy()

    return img_out


class agg_segmentation_wrapper_module(nn.Module):

    def __init__(self, model):
        super(agg_segmentation_wrapper_module, self).__init__()
        self._model = model

    def forward(self, x):

        model_out = self._model(x)


        out_max = torch.argmax(model_out, dim=1, keepdim=True)
        selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)

        return (model_out * selected_inds).sum(dim=(2,3))



def read_data_to_predict(path_to_data_to_predict: str):
    data = tiff.imread(path_to_data_to_predict)
    label = data[3, :, :]
    image = data[:3, :, :]
    return image, label


def write_results(results_array: np.ndarray, path_to_write_to) -> None:
    """
    Writes the output into a file.
    :param results_array: output as a numpy array
    :param path_to_write_to: Output path
    """
    os.makedirs(pathlib.Path(path_to_write_to).parent.absolute(), exist_ok=True)
    np.save(path_to_write_to, results_array)
    pass

def mask_binning(classification: torch.Tensor):
    classification = np.argmax(classification, axis=0)
    return classification


def write_ome_out(image, classification, out_name) -> None:
    full_image = np.zeros((256, 256, 4))
    full_image[:, :, 0] = image[0, :, :]
    full_image[:, :, 1] = image[1, :, :]
    full_image[:, :, 2] = image[2, :, :]
    full_image[:, :, 3] = mask_binning(classification[:, :, :])
    full_image = np.transpose(full_image, (2, 0, 1))
    with tiff.TiffWriter(os.path.join(".", out_name), bigtiff=True) as tif_file:
        metadata = {"axes": "CYX",
                    'Channel': {"Name": ["red", "green", "blue", "mask"]}}
        tif_file.write(full_image, photometric="rgb", metadata=metadata)


def get_pytorch_model(path_to_pytorch_model: str, sanitize: bool, architecture: str):

    if not _check_exists(path_to_pytorch_model):
        download(architecture)

    if architecture == "U-Net":
        model = Unet(len_test_set=128, hparams={}, input_channels=3, num_classes=7, flat_weights=True, dropout_val=True)
        model.apply(weights_init)
        state_dict = torch.load("models/CU_NET.ckpt", map_location="cpu")
    elif architecture == "CU-Net":
        model = ContextUnet(len_test_set=128, hparams={}, input_channels=3, num_classes=7, flat_weights=True, dropout_val=True)
        model.apply(weights_init)
        state_dict = torch.load("models/CU_NET.ckpt", map_location="cpu")
    else:
        raise KeyError("Architecture not available")
    
    model.load_state_dict(state_dict["state_dict"], strict=False)
    model.eval()

    if sanitize:
        os.remove(path_to_pytorch_model)

    return model


def _check_exists(filepath) -> bool:
    return os.path.exists(filepath)


def download(architecture) -> None:
    """Download the model if it doesn't exist in processed_folder already."""

    mirrors = [
        'https://zenodo.org/record/',
    ]

    if architecture == "U-Net":
        resources = [
            ("U_NET.ckpt", "7884684/files/U_NET.ckpt", "17511a0af673df264179fb93d73c9dd5"),
        ]
    elif architecture == "CU-Net":
        resources = [
            ("CU_NET.ckpt", "7884684/files/CU_NET.ckpt", "9090252a639c39c9f9509df7e1ce311c"),
        ]
    else:
        raise IOError("No architecture found")
    
    # download files
    for filename, uniqueID, md5 in resources:
        for mirror in mirrors:
            url = "{}{}".format(mirror, uniqueID)
            try:
                print("Downloading {}".format(url))
                download_url(
                    url, root=str(pathlib.Path("models/" + architecture).parent.absolute()),
                    filename=filename,
                    md5=md5
                )
            except URLError as error:
                print(
                    "Failed to download (trying next):\n{}".format(error)
                )
                continue
            finally:
                print()
            break
        else:
            raise RuntimeError("Error downloading {}".format(filename))
    print('Done!')



if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover