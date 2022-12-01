
# Download and inspect the data

import fiftyone as fo
import fiftyone.zoo as foz

#Here you can choose your dataset, the spit, the labels etc. For more 
#information please visit fiftyone documentation here -> https://voxel51.com/docs/fiftyone/

dataset = foz.load_zoo_dataset(
              "open-images-v6",
              split="validation",
              label_types=["detections"],
              classes=["Cat"],
              max_samples=2,
          )

#Here is a way on how to actually save the data in a specific folder
#By adjusting the dataset_type you can download data in specific formats
#ready for models implementation. 


export_dir = "{path}"
dataset_type=fo.types.FiftyOneImageDetectionDataset
dataset.export(export_dir=export_dir,dataset_type=dataset_type)



#Here is a way to open a browser page and to inspect the data
#in real-time


session = fo.launch_app(dataset, port=5151)
session.wait()
