import argparse, os, yaml
import numpy as np

from img_randomer import transforms
from img_randomer import Image_randomer
from PIL import Image
from pathlib import Path
class Image_loader:
    def __init__(self, config):
        self._config = config
        self.path = config.get("datapath","data/pnp_30_ep/pick_and_place")
        self.randomer = Image_randomer(config)

        
    def get_episode_path(self):
        dataset_path = Path(self.path)
        episode_paths = []
        components = os.listdir(dataset_path)
        components.sort()  # 给所有内容排序
        for role in components:
            role_path = os.path.join(dataset_path, role)
            if os.path.isdir(role_path): 
                # print(f"find role path {role_path}")
                episode_paths.append(role_path)
        return episode_paths
     
    def get_color_path(self):
        color_paths = []
        episode_paths = self.get_episode_path()
        for episode_path in episode_paths:
            color_path = os.path.join(episode_path,"colors")
            if os.path.isdir(color_path):
                color_paths.append(color_path)
            else:
                raise ValueError(f"color path {color_path} is not a dir")
        return color_paths

    def get_img_dict(self):
        img_dict = {}
        color_paths = self.get_color_path()
        for color_path in color_paths:
            episode_path = os.path.dirname(color_path)
            episode_name = os.path.basename(episode_path)
            img_dict[episode_name] = {}    
            image_components = os.listdir(color_path)
            image_components.sort()
            for img_name in image_components:
                img_path = os.path.join(color_path,img_name)
                if os.path.isfile(img_path):
                    img_key = os.path.splitext(img_name)[0]
                    img_ext = os.path.splitext(img_name)[1]
                    if img_ext == ".jpg":
                        # print(f"color ext: {img_ext}")
                        img_dict[episode_name][img_key] = img_path
                    else:
                        print("????????????????????????????/n")
                        print(f"color ext: {img_ext}")
                        continue
        return img_dict

    def load_image(self):
        image_dict = self.get_img_dict()
        for episode_idx in image_dict:
            for image_key in image_dict[episode_idx]:
                image_path = image_dict[episode_idx][image_key]
                img = Image.open(image_path)

                rgb_img = img.convert("RGB")               
                image_tensor = self.randomer(rgb_img)
                return image_tensor


    # 用transforms.ToPILImage方法反归一化图片然后show,save
    # 考虑在层中间正常可视化
    def show_tensor_image(self, image_tensor):
        # image = image.tensor.permute(1, 2, 0)
        tensor_to_pil = transforms.ToPILImage()
        pil_image = tensor_to_pil(image_tensor)  
        # pil_image.show()
        pil_image.save("debug_augmented.jpg")
        return pil_image


if __name__ == "__main__":

    arguments = {"config": {"short_cut": "-c",
                        "symbol": "--config",
                        "type": str,
                        "default": "config/img.yaml",
                        "help": "Path to the config file"}}
    args = argparse.ArgumentParser("img loader",arguments)
    for arg_name, arg_info in arguments.items():
        args.add_argument(arg_info["short_cut"], arg_info["symbol"], type=arg_info["type"], default=arg_info["default"], help=arg_info["help"])
    args = args.parse_args()
    config = yaml.safe_load(open(args.config, "r"))


    image_loader = Image_loader(
        config = config
    )
    # print(img_dict)
    image = image_loader.load_image()
    image_loader.show_tensor_image(image)
